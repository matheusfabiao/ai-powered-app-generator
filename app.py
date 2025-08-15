import json
import os
import socket  # Needed to find an open network port for the preview
import subprocess  # Needed to run other Streamlit apps (the preview)
import sys  # Needed to get the path to the current Python executable
import time
from pathlib import Path

import google.generativeai as genai
import streamlit as st
import streamlit_antd_components as sac  # Using for specific buttons (Save/Delete group)
from dotenv import load_dotenv
from streamlit_ace import st_ace

# --- UI Components ---
# These libraries provide pre-built UI elements like menus and the code editor.
from streamlit_option_menu import option_menu

# --- Configuration ---
st.set_page_config(
    layout='wide',
    page_title='Gerador de Apps IA',
    page_icon='🤖',
)
load_dotenv()   # Load API keys from a file named .env in the same directory

# --- Constants ---
# Where generated Python app files will be saved
WORKSPACE_DIR = Path('workspace_st_apps')
WORKSPACE_DIR.mkdir(exist_ok=True)   # Create the directory if it doesn't exist

# Code editor appearance settings
ACE_DEFAULT_THEME = 'monokai'
ACE_DEFAULT_KEYBINDING = 'vscode'

# Which Google AI model to use for generating code
GEMINI_MODEL_NAME = os.getenv(
    'GOOGLE_AI_MODEL', 'gemini-2.5-pro-preview-06-05'
)

# Instructions for the Google AI model
# This tells the AI how to format its responses (as JSON commands)
GEMINI_SYSTEM_PROMPT = f"""
Você é um assistente de IA que ajuda a criar aplicativos Streamlit.
Seu objetivo é gerenciar arquivos Python em um espaço de trabalho com base nas solicitações dos usuários.
Responda *apenas* com um array JSON válida contendo comandos. Não adicione nenhuma explicação antes ou depois do array JSON.

Comandos disponíveis:
1.  `{{"action": "create_update", "filename": "app_name.py", "content": "FULL_PYTHON_CODE_HERE"}}`
    - Use isso para criar um novo arquivo Python ou sobrescrever completamente um arquivo existente.
    - Forneça o conteúdo *completo* do arquivo. Escape as barras invertidas (`\\\\`) e as aspas duplas (`\\"`). Certifique-se de que as novas linhas sejam `\\n`.
    - Não inclua blocos de marcação Python ou shebangs (`#!/usr/bin/env python`) no “content”.
2.  `{{"action": "delete", "filename": "old_app.py"}}`
    - Use isso para excluir um arquivo Python da área de trabalho.
3.  `{{"action": "chat", "content": "Your message here."}}`
    - Use isso *somente* se precisar pedir esclarecimentos, relatar um problema que não consegue resolver com ações de arquivo ou confirmar o entendimento.

Arquivos Python atuais na área de trabalho: {', '.join([f.name for f in WORKSPACE_DIR.iterdir() if f.is_file() and f.suffix == '.py']) if WORKSPACE_DIR.exists() else 'None'}

Exemplo de interação:
Usuário: Crie um aplicativo simples chamado ola_mundo.py.
AI: `[{{"action": "create_update", "filename": "hello.py", "content": "import streamlit as st\\n\\nst.title('Olá Mundo!')\\nst.write('Este é um aplicativo simples.')"}}`

Certifique-se de que toda a sua resposta seja *apenas* o array JSON. `[...]`.
"""

# --- API Client Setup ---
try:
    google_api_key = os.getenv('GOOGLE_API_KEY')
    if not google_api_key:
        # Stop the app if the API key is missing
        st.error(
            '🔴 Chave da API do Google não encontrada. Por favor, defina `GOOGLE_API_KEY` em um arquivo `.env`.'
        )
        st.stop()   # Halt execution
    # Configure the Gemini library with the key
    genai.configure(api_key=google_api_key)
    # Create the AI model object
    model = genai.GenerativeModel(GEMINI_MODEL_NAME)
except Exception as e:
    st.error(f'🔴 Falha ao configurar Google AI: {e}')
    st.stop()

# --- Session State ---
# Streamlit reruns the script on interaction. Session state stores data
# between reruns, like chat history or which file is selected.
def initialize_session_state():
    """Sets up default values in Streamlit's session state dictionary."""
    state_defaults = {
        'messages': [],  # List to store chat messages (user and AI)
        'selected_file': None,  # Name of the file currently shown in the editor
        'file_content_on_load': '',  # Content of the selected file when loaded (read-only)
        'preview_process': None,  # Stores the running preview process object
        'preview_port': None,  # Port number used by the preview
        'preview_url': None,  # URL to access the preview
        'preview_file': None,  # Name of the file being previewed
        'editor_unsaved_content': '',  # Current text typed into the editor
        'last_saved_content': '',  # Content that was last successfully saved to disk
    }
    for key, default_value in state_defaults.items():
        if key not in st.session_state:
            st.session_state[key] = default_value


initialize_session_state()   # Run the initialization

# --- File System Functions ---
def get_workspace_python_files():
    """Gets a list of all '.py' filenames in the workspace directory."""
    if not WORKSPACE_DIR.is_dir():
        return []   # Return empty list if directory doesn't exist
    try:
        # List files, filter for .py, sort alphabetically
        python_files = sorted(
            [
                f.name
                for f in WORKSPACE_DIR.iterdir()
                if f.is_file() and f.suffix == '.py'
            ]
        )
        return python_files
    except Exception as e:
        st.error(f'Erro ao ler diretório do workspace: {e}')
        return []


def read_file(filename):
    """Reads the text content of a file from the workspace."""
    if not filename:   # Check if filename is provided
        return None
    # Prevent accessing files outside the workspace (basic security)
    if '..' in filename or filename.startswith(('/', '\\')):
        st.error(f'Caminho de arquivo inválido: {filename}')
        return None

    filepath = WORKSPACE_DIR / filename   # Combine directory and filename
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            return f.read()   # Return the file's text content
    except FileNotFoundError:
        st.warning(f'Arquivo não encontrado: {filename}')
        return None   # Indicate file doesn't exist
    except Exception as e:
        st.error(f"Erro ao ler arquivo '{filename}': {e}")
        return None


def save_file(filename, content):
    """Writes text content to a file in the workspace."""
    if not filename:
        return False   # Cannot save without a filename
    if '..' in filename or filename.startswith(('/', '\\')):
        st.error(f'Caminho de arquivo inválido: {filename}')
        return False

    filepath = WORKSPACE_DIR / filename
    try:
        # Write the content to the file (overwrites if it exists)
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(content)
        return True   # Indicate success
    except Exception as e:
        st.error(f"Erro ao salvar arquivo '{filename}': {e}")
        return False   # Indicate failure


def delete_file(filename):
    """Deletes a file from the workspace and updates app state."""
    if not filename:
        return False
    if '..' in filename or filename.startswith(('/', '\\')):
        st.error(f'Caminho de arquivo inválido: {filename}')
        return False

    filepath = WORKSPACE_DIR / filename
    try:
        if filepath.is_file():
            os.remove(filepath)   # Delete the actual file
            st.toast(f'Excluído: {filename}', icon='🗑️')

            # If the deleted file was being previewed, stop the preview
            if st.session_state.preview_file == filename:
                stop_preview()   # Call the function to stop the process

            # If the deleted file was selected in the editor, clear the selection
            if st.session_state.selected_file == filename:
                st.session_state.selected_file = None
                st.session_state.file_content_on_load = ''
                st.session_state.editor_unsaved_content = ''
                st.session_state.last_saved_content = ''
            return True   # Indicate success
        else:
            st.warning(
                f"Não foi possível excluir: Arquivo '{filename}' não encontrado."
            )
            return False
    except Exception as e:
        st.error(f"Erro ao excluir arquivo '{filename}': {e}")
        return False


# --- AI Interaction Functions ---


def _clean_ai_response_text(ai_response_text):
    """Removes potential code fences (```json ... ```) from AI response."""
    text = ai_response_text.strip()
    if text.startswith('```json'):
        text = text[7:-3].strip()   # Remove ```json and ```
    elif text.startswith('```'):
        text = text[3:-3].strip()   # Remove ``` and ```
    return text


def parse_and_execute_ai_commands(ai_response_text):
    """
    Parses the AI's JSON response and performs the requested file actions.
    Returns the list of commands (for chat history display).
    """
    cleaned_text = _clean_ai_response_text(ai_response_text)
    executed_commands_list = []   # To store commands for chat display

    try:
        # Attempt to convert the cleaned text into a Python list of dictionaries
        commands = json.loads(cleaned_text)

        # Check if the result is actually a list
        if not isinstance(commands, list):
            st.error(
                'A resposta da IA foi JSON válido, mas não uma lista de comandos.'
            )
            # Return a chat message indicating the error for display
            return [
                {
                    'action': 'chat',
                    'content': f'AI Error: Response was not a list. Response: {cleaned_text}',
                }
            ]

        # Process each command dictionary in the list
        for command_data in commands:
            # Ensure the command is a dictionary before processing
            if not isinstance(command_data, dict):
                st.warning(
                    f'A IA enviou um formato de comando inválido (não é um dict): {command_data}'
                )
                executed_commands_list.append(
                    {
                        'action': 'chat',
                        'content': f'AI Error: Invalid command format: {command_data}',
                    }
                )
                continue   # Skip to the next command

            # Add the command to the list we return (used for displaying AI actions)
            executed_commands_list.append(command_data)

            # Get action details from the dictionary
            action = command_data.get('action')
            filename = command_data.get('filename')
            content = command_data.get('content')

            # --- Execute the action ---
            if action == 'create_update':
                if filename and content is not None:
                    success = save_file(filename, content)
                    if success:
                        st.toast(f'IA salvou: {filename}', icon='💾')
                        # If this file is currently open in the editor, update the editor's content
                        if st.session_state.selected_file == filename:
                            st.session_state.file_content_on_load = content
                            st.session_state.last_saved_content = content
                            st.session_state.editor_unsaved_content = content
                    else:
                        st.error(
                            f"Falha no comando da IA: Não foi possível salvar '{filename}'."
                        )
                        # Add error details to chat display list
                        executed_commands_list.append(
                            {
                                'action': 'chat',
                                'content': f'Error: Failed saving {filename}',
                            }
                        )
                else:
                    st.warning(
                        "Comando 'create_update' da IA sem nome de arquivo ou conteúdo."
                    )
                    executed_commands_list.append(
                        {
                            'action': 'chat',
                            'content': 'AI Warning: Invalid create_update',
                        }
                    )

            elif action == 'delete':
                if filename:
                    success = delete_file(filename)
                    if not success:
                        st.error(
                            f"Falha no comando da IA: Não foi possível excluir '{filename}'."
                        )
                        executed_commands_list.append(
                            {
                                'action': 'chat',
                                'content': f'Error: Failed deleting {filename}',
                            }
                        )
                else:
                    st.warning("Comando 'delete' da IA sem nome de arquivo.")
                    executed_commands_list.append(
                        {
                            'action': 'chat',
                            'content': 'AI Warning: Invalid delete',
                        }
                    )

            elif action == 'chat':
                # No action needed here, the chat message is already in executed_commands_list
                # and will be displayed in the chat history.
                pass

            else:
                # Handle unrecognized actions from the AI
                st.warning(f"A IA enviou ação desconhecida: '{action}'.")
                executed_commands_list.append(
                    {
                        'action': 'chat',
                        'content': f"AI Warning: Unknown action '{action}'",
                    }
                )

        return executed_commands_list   # Return the list for chat display

    except json.JSONDecodeError:
        st.error(
            f'A resposta da IA não foi JSON válido.\nResposta bruta:\n```\n{cleaned_text}\n```'
        )
        # Return a chat message indicating the JSON error for display
        return [
            {
                'action': 'chat',
                'content': f'AI Error: Invalid JSON received. Response: {ai_response_text}',
            }
        ]
    except Exception as e:
        st.error(f'Erro ao processar comandos da IA: {e}')
        return [
            {'action': 'chat', 'content': f'Error processing commands: {e}'}
        ]


def _prepare_gemini_history(chat_history, system_prompt):
    """Formats chat history for the Gemini API call."""
    gemini_history = []
    # Start with the system prompt (instructions for the AI)
    gemini_history.append({'role': 'user', 'parts': [{'text': system_prompt}]})
    # Gemini requires a model response to start the turn properly after a system prompt
    gemini_history.append(
        {
            'role': 'model',
            'parts': [
                {
                    'text': json.dumps(
                        [
                            {
                                'action': 'chat',
                                'content': 'Understood. I will respond only with JSON commands.',
                            }
                        ]
                    )
                }
            ],
        }
    )

    # Add the actual user/assistant messages from session state
    for msg in chat_history:
        role = msg['role']   # "user" or "assistant"
        content = msg['content']
        api_role = (
            'model' if role == 'assistant' else 'user'
        )   # Map to API roles

        # Convert assistant messages (which are lists of commands) back to JSON strings
        if role == 'assistant' and isinstance(content, list):
            try:
                content_str = json.dumps(content)
            except Exception:
                content_str = str(content)   # Fallback if conversion fails
        else:
            content_str = str(content)   # User messages are already strings

        if content_str:   # Avoid sending empty messages
            gemini_history.append(
                {'role': api_role, 'parts': [{'text': content_str}]}
            )

    return gemini_history


def ask_gemini_ai(chat_history):
    """Sends the conversation history to the Gemini AI and returns its response."""

    # Get current list of files to include in the prompt context
    current_files = get_workspace_python_files()
    file_list_info = f"Current Python files: {', '.join(current_files) if current_files else 'None'}"
    # Update the system prompt with the current file list
    updated_system_prompt = GEMINI_SYSTEM_PROMPT.replace(
        'Current Python files: ...',  # Placeholder text to replace
        file_list_info,
    )

    # Prepare the history in the format the API expects
    gemini_api_history = _prepare_gemini_history(
        chat_history, updated_system_prompt
    )

    try:
        # Make the API call to Google
        # print(f"DEBUG: Sending history:\n{json.dumps(gemini_api_history, indent=2)}") # Uncomment for debugging API calls
        response = model.generate_content(gemini_api_history)
        # print(f"DEBUG: Received response:\n{response.text}") # Uncomment for debugging API calls
        return response.text   # Return the AI's raw text response

    except Exception as e:
        # Handle potential errors during the API call
        error_message = f'Gemini API call failed: {type(e).__name__}'
        st.error(f'🔴 {error_message}: {e}')

        # Try to give a more user-friendly error message for common issues
        error_content = f'Erro da IA: {str(e)[:150]}...'   # Default message
        if 'API key not valid' in str(e):
            error_content = 'Erro da IA: Chave da API do Google inválida.'
        elif (
            '429' in str(e)
            or 'quota' in str(e).lower()
            or 'resource has been exhausted' in str(e).lower()
        ):
            error_content = (
                'Erro da IA: Cota ou Limite de Taxa da API Excedido.'
            )
        # Handle cases where the AI's response might be blocked for safety
        try:
            if (
                response
                and response.prompt_feedback
                and response.prompt_feedback.block_reason
            ):
                error_content = f'Erro da IA: Entrada bloqueada por filtros de segurança ({response.prompt_feedback.block_reason}).'
            elif (
                response
                and response.candidates
                and response.candidates[0].finish_reason != 'STOP'
            ):
                error_content = f'Erro da IA: Resposta parada ({response.candidates[0].finish_reason}). Pode ser devido a filtros de segurança ou limites de tamanho.'
        except Exception:
            pass   # Ignore errors during safety check parsing

        # Return the error as a JSON chat command so it appears in the chat history
        return json.dumps([{'action': 'chat', 'content': error_content}])


# --- Live Preview Process Management ---
def _find_available_port():
    """Finds an unused network port."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(('', 0))   # Bind to port 0 to let the OS choose a free port
        return s.getsockname()[1]   # Return the chosen port number


def stop_preview():
    """Stops the currently running Streamlit preview process."""
    process_to_stop = st.session_state.get('preview_process')
    pid = getattr(process_to_stop, 'pid', None)   # Get process ID if available

    if process_to_stop and pid:
        st.info(f'Parando processo de visualização (PID: {pid})...')
        try:
            # Check if the process is still running
            if process_to_stop.poll() is None:
                # Ask the process to terminate gracefully
                process_to_stop.terminate()
                try:
                    # Wait up to 3 seconds for it to close
                    process_to_stop.wait(timeout=3)
                    st.toast(
                        f'Processo de visualização {pid} parado.', icon='⏹️'
                    )
                except subprocess.TimeoutExpired:
                    # If it didn't stop, force kill it
                    st.warning(
                        f'Processo de visualização {pid} não parou normalmente, matando...'
                    )
                    if (
                        process_to_stop.poll() is None
                    ):   # Check again before kill
                        process_to_stop.kill()
                        process_to_stop.wait(timeout=1)   # Brief wait for kill
                        st.toast(
                            f'Processo de visualização {pid} morto.', icon='💀'
                        )
            else:
                # Process was already finished
                st.warning(f'Processo de visualização {pid} já havia parado.')
        except ProcessLookupError:
            st.warning(
                f'Processo de visualização {pid} não encontrado (já foi embora?).'
            )
        except Exception as e:
            st.error(
                f'Erro ao tentar parar processo de visualização {pid}: {e}'
            )

    # Always clear the preview state variables after attempting to stop
    st.session_state.preview_process = None
    st.session_state.preview_port = None
    st.session_state.preview_url = None
    st.session_state.preview_file = None
    st.rerun()   # Update the UI immediately


def start_preview(python_filename):
    """Starts a Streamlit app preview in a separate process."""
    filepath = WORKSPACE_DIR / python_filename
    # Basic check: ensure the file exists and is a Python file
    if not filepath.is_file() or filepath.suffix != '.py':
        st.error(
            f"Não é possível visualizar: '{python_filename}' não é um arquivo Python válido."
        )
        return False

    # Stop any currently running preview first
    if st.session_state.get('preview_process'):
        st.warning('Parando visualização existente primeiro...')
        stop_preview()   # This function will rerun, so we might need to adjust flow
        # Let's add a small delay here AFTER stop_preview (which reruns) handles its part.
        # This might mean the button needs to be clicked twice sometimes, but simplifies state.
        # A more complex approach would involve flags in session state.
        time.sleep(0.5)   # Brief pause

    with st.spinner(f'Iniciando visualização para `{python_filename}`...'):
        try:
            port = _find_available_port()
            # Command to run: python -m streamlit run <filepath> --port <port> [options]
            command = [
                sys.executable,  # Use the same Python interpreter running this script
                '-m',
                'streamlit',
                'run',
                str(filepath.resolve()),  # Use the full path to the file
                '--server.port',
                str(port),
                '--server.headless',
                'true',  # Don't open a browser automatically
                '--server.runOnSave',
                'false',  # Don't automatically rerun on save
                '--server.fileWatcherType',
                'none',  # Don't watch for file changes
            ]

            # Start the command as a new process
            preview_proc = subprocess.Popen(
                command,
                stdout=subprocess.PIPE,  # Capture output (optional)
                stderr=subprocess.PIPE,  # Capture errors
                text=True,
                encoding='utf-8',
            )

            # Give Streamlit a moment to start up or fail
            time.sleep(2.5)   # Wait a bit

            # Check if the process started successfully (is still running)
            if preview_proc.poll() is None:
                # Success! Store process info in session state
                st.session_state.preview_process = preview_proc
                st.session_state.preview_port = port
                st.session_state.preview_url = f'http://localhost:{port}'
                st.session_state.preview_file = python_filename
                st.success(
                    f'Visualização iniciada: {st.session_state.preview_url}'
                )
                st.toast(
                    f'Visualização em execução para {python_filename}',
                    icon='🚀',
                )
                return True
            else:
                # Failure: Process ended quickly, likely an error
                st.error(
                    f'Falha ao iniciar visualização para `{python_filename}`.'
                )
                # Try to show error output from the failed process
                try:
                    stderr_output = preview_proc.stderr.read()
                    if stderr_output:
                        st.error('Saída de Erro da Visualização:')
                        st.code(stderr_output, language=None)
                    else:   # If no stderr, maybe there was stdout?
                        stdout_output = preview_proc.stdout.read()
                        if stdout_output:
                            st.error(
                                'Saída da Visualização (pode conter erros):'
                            )
                            st.code(stdout_output, language=None)
                except Exception as read_e:
                    st.error(
                        f'Não foi possível ler saída do processo de visualização falhado: {read_e}'
                    )
                # Clear any partial state
                st.session_state.preview_process = None
                return False
        except Exception as e:
            st.error(f'Erro ao tentar iniciar processo de visualização: {e}')
            st.session_state.preview_process = None   # Ensure clean state
            return False


# --- Streamlit App UI ---

st.title('🤖 Gerador de Apps com IA')

# --- Sidebar ---
with st.sidebar:
    # Logo
    st.image('img/Logo Branca.webp', width=100)

    st.header('💬 Chat & Controles')
    st.divider()

    # --- Chat History Display ---
    chat_container = st.container(height=400)
    with chat_container:
        if not st.session_state.messages:
            st.info('Histórico de chat vazio. Digite suas instruções abaixo.')
        else:
            # Loop through messages stored in session state
            for message in st.session_state.messages:
                role = message['role']   # "user" or "assistant"
                content = message['content']
                avatar = '🧑‍💻' if role == 'user' else '🤖'

                # Display message using Streamlit's chat message element
                with st.chat_message(role, avatar=avatar):
                    if role == 'assistant' and isinstance(content, list):
                        # Assistant message contains commands - format them nicely
                        file_actions_summary = ''
                        chat_responses = []
                        code_snippets = []

                        for command in content:
                            if not isinstance(command, dict):
                                continue   # Skip malformed

                            action = command.get('action')
                            filename = command.get('filename')
                            cmd_content = command.get('content')

                            if action == 'create_update':
                                file_actions_summary += (
                                    f'📝 **Salvo:** `{filename}`\n'
                                )
                                if cmd_content:
                                    code_snippets.append(
                                        {
                                            'filename': filename,
                                            'content': cmd_content,
                                        }
                                    )
                            elif action == 'delete':
                                file_actions_summary += (
                                    f'🗑️ **Excluído:** `{filename}`\n'
                                )
                            elif action == 'chat':
                                chat_responses.append(
                                    str(cmd_content or '...')
                                )
                            else:
                                file_actions_summary += (
                                    f'⚠️ **Ação Desconhecida:** `{action}`\n'
                                )

                        # Display the formatted summary and chat responses
                        full_display_text = (
                            file_actions_summary + '\n'.join(chat_responses)
                        ).strip()
                        if full_display_text:
                            st.markdown(full_display_text)
                        else:   # Handle cases where AI might return empty actions
                            st.markdown('(A IA não realizou ações exibíveis)')

                        # Show code snippets in collapsible sections
                        for snippet in code_snippets:
                            with st.expander(
                                f"Ver Código para `{snippet['filename']}`",
                                expanded=False,
                            ):
                                st.code(snippet['content'], language='python')

                    elif isinstance(content, str):
                        # Simple text message (from user or AI chat action)
                        st.write(content)
                    else:
                        # Fallback for unexpected content type
                        st.write(f'Unexpected message format: {content}')

    # --- Chat Input Box ---
    user_prompt = st.chat_input('Converse com a IA...')
    if user_prompt:
        # 1. Add user's message to the chat history (in session state)
        st.session_state.messages.append(
            {'role': 'user', 'content': user_prompt}
        )

        # 2. Show a spinner while waiting for the AI
        with st.spinner('🧠 IA Pensando...'):
            # 3. Send the *entire* chat history to the AI
            ai_response_text = ask_gemini_ai(st.session_state.messages)
            # 4. Parse the AI's response and execute file commands
            ai_commands_executed = parse_and_execute_ai_commands(
                ai_response_text
            )

        # 5. Add the AI's response (the list of executed commands) to chat history
        st.session_state.messages.append(
            {'role': 'assistant', 'content': ai_commands_executed}
        )

        # 6. Rerun the script immediately to show the new messages and update file list/editor
        st.rerun()

    st.divider()

    # --- Status Info ---
    st.subheader('💡 Status & Informações')
    st.success(f'Usando modelo de IA: {GEMINI_MODEL_NAME}', icon='✅')
    st.warning(
        '**Nota:** Revise o código da IA antes de executar visualizações. `create_update` sobrescreve arquivos.',
    )

    st.divider()

    # --- Footer ---
    st.write(
        'Desenvolvido com ❤️ por:\n[Matheus Fabião](https://github.com/matheusfabiao)',
        unsafe_allow_html=True,
    )

# --- Main Area Tabs ---
selected_tab = option_menu(
    menu_title=None,
    options=['Workspace', 'Visualização ao Vivo'],
    icons=['folder-fill', 'play-btn-fill'],
    orientation='horizontal',
    key='main_tab_menu'
    # Removed custom styles for simplicity
)

# --- Workspace Tab ---
if selected_tab == 'Workspace':
    st.header('📂 Workspace & Editor')
    st.divider()

    # Create two columns: one for file list, one for editor
    file_list_col, editor_col = st.columns(
        [0.3, 0.7]
    )   # 30% width for files, 70% for editor

    with file_list_col:
        st.subheader('Arquivos')
        python_files = get_workspace_python_files()

        # Prepare options for the dropdown menu
        select_options = ['--- Select a file ---'] + python_files
        current_selection_in_state = st.session_state.get('selected_file')

        # Find the index of the currently selected file to set the dropdown default
        try:
            current_index = (
                select_options.index(current_selection_in_state)
                if current_selection_in_state
                else 0
            )
        except ValueError:
            current_index = (
                0  # If file in state doesn't exist, default to "Select"
            )

        # The dropdown widget
        selected_option = st.selectbox(
            'Editar arquivo:',
            options=select_options,
            index=current_index,
            key='file_selector_dropdown',
            label_visibility='collapsed',  # Hide the label "Edit file:"
        )

        # --- Handle File Selection Change ---
        # If the dropdown selection is different from what's stored in session state...
        newly_selected_filename = (
            selected_option
            if selected_option != '--- Select a file ---'
            else None
        )
        if newly_selected_filename != current_selection_in_state:
            st.session_state.selected_file = (
                newly_selected_filename  # Update state
            )
            # Read the content of the newly selected file
            file_content = (
                read_file(newly_selected_filename)
                if newly_selected_filename
                else ''
            )
            # Handle case where file read failed (e.g., it was deleted)
            if file_content is None and newly_selected_filename:
                file_content = (
                    f"# ERROR: Could not read file '{newly_selected_filename}'"
                )

            # Update session state with the file's content for the editor
            st.session_state.file_content_on_load = file_content
            st.session_state.editor_unsaved_content = (
                file_content  # Start editor with file content
            )
            st.session_state.last_saved_content = (
                file_content  # Mark as saved initially
            )
            st.rerun()   # Rerun script to load the new file into the editor

    with editor_col:
        st.subheader('Editor de Código')
        selected_filename = st.session_state.selected_file

        if selected_filename:
            st.caption(f'Editando: `{selected_filename}`')

            # Display the Ace code editor widget
            editor_current_text = st_ace(
                value=st.session_state.get(
                    'editor_unsaved_content', ''
                ),  # Show unsaved content
                language='python',
                theme=ACE_DEFAULT_THEME,
                keybinding=ACE_DEFAULT_KEYBINDING,
                font_size=14,
                tab_size=4,
                wrap=True,
                auto_update=False,  # Don't trigger reruns on every keystroke
                key=f'ace_editor_{selected_filename}',  # Unique key helps reset state on file change
            )

            # Check if the editor's current text is different from the last saved text
            has_unsaved_changes = (
                editor_current_text != st.session_state.last_saved_content
            )

            # If the text in the editor box changes, update our 'unsaved' state variable
            if editor_current_text != st.session_state.editor_unsaved_content:
                st.session_state.editor_unsaved_content = editor_current_text
                st.rerun()   # Rerun to update the 'Save Changes' button state

            # --- Editor Action Buttons ---
            # Using sac.buttons here for the nice grouped layout with icons.
            editor_buttons = [
                sac.ButtonsItem(
                    label='💾 Salvar Alterações',
                    icon='save',
                    disabled=not has_unsaved_changes,
                ),
                sac.ButtonsItem(
                    label='🗑️ Excluir Arquivo', icon='trash', color='red'
                ),
            ]
            clicked_editor_button = sac.buttons(
                items=editor_buttons,
                index=None,
                format_func='title',
                align='end',
                size='small',
                return_index=False,
                key='editor_action_buttons',
            )

            # --- Handle Button Clicks ---
            if clicked_editor_button == '💾 Salvar Alterações':
                if save_file(selected_filename, editor_current_text):
                    # Update state to reflect the save
                    st.session_state.file_content_on_load = editor_current_text
                    st.session_state.last_saved_content = editor_current_text
                    st.toast(f'Salvo: `{selected_filename}`', icon='💾')
                    time.sleep(0.5)   # Let toast message show
                    st.rerun()   # Rerun to disable the save button
                else:
                    st.error(
                        f"Erro: Não foi possível salvar '{selected_filename}'."
                    )

            elif clicked_editor_button == '🗑️ Excluir Arquivo':
                # Use sac.confirm_button for a confirmation pop-up
                needs_confirmation = True   # Flag to show confirmation
                if needs_confirmation:
                    confirmed = sac.confirm_button(
                        f'Excluir `{selected_filename}`?',  # Confirmation message
                        color='error',
                        key='confirm_delete_button',
                    )
                    if confirmed:
                        if delete_file(selected_filename):
                            # Deletion successful, file list and editor will update on rerun
                            st.rerun()
                        # No 'else' needed, delete_file shows errors

            # Show a warning if there are unsaved changes
            if has_unsaved_changes:
                st.warning('Você tem alterações não salvas.')

        else:
            # Show a placeholder message if no file is selected
            st.info(
                'Selecione um arquivo Python da lista à esquerda para visualizar ou editar.'
            )
            st_ace(
                value='# Selecione um arquivo...',
                language='python',
                readonly=True,
                key='ace_placeholder',
            )

# --- Live Preview Tab ---
elif selected_tab == 'Visualização ao Vivo':
    st.header('▶️ Visualização ao Vivo')
    st.divider()
    st.warning(
        '⚠️ Executar código gerado pela IA pode ter consequências não intencionais. Revise o código primeiro!'
    )

    # Get preview status from session state
    is_preview_running = st.session_state.get('preview_process') is not None
    file_being_previewed = st.session_state.get('preview_file')
    preview_url = st.session_state.get('preview_url')
    selected_file_for_preview = st.session_state.get(
        'selected_file'
    )   # File selected in Workspace

    # --- Preview Controls ---
    st.subheader('Controles')
    if not selected_file_for_preview:
        st.info(
            "Selecione um arquivo na aba 'Workspace' para habilitar os controles de visualização."
        )
        # Allow stopping a preview even if no file is selected
        if is_preview_running:
            st.warning(
                f'Visualização em execução para: `{file_being_previewed}`'
            )
            if st.button(
                f'⏹️ Parar Visualização ({file_being_previewed})',
                key='stop_other_preview',
            ):
                stop_preview()   # Will stop and rerun
    else:
        # Controls for the file selected in the Workspace
        st.write(
            f'Arquivo selecionado para visualização: `{selected_file_for_preview}`'
        )
        is_python = selected_file_for_preview.endswith('.py')

        if not is_python:
            st.error(
                'Não é possível visualizar: Arquivo selecionado não é um arquivo Python (.py).'
            )
        else:
            # Layout Run and Stop buttons side-by-side
            run_col, stop_col = st.columns(2)
            with run_col:
                # Disable Run button if a preview is already running
                run_disabled = is_preview_running
                if st.button(
                    '🚀 Executar Visualização',
                    disabled=run_disabled,
                    type='primary',
                    use_container_width=True,
                ):
                    if start_preview(selected_file_for_preview):
                        st.rerun()   # Rerun to show the preview iframe
            with stop_col:
                # Disable Stop button if no preview is running OR if the running preview
                # is for a DIFFERENT file than the one currently selected in the workspace.
                stop_disabled = not is_preview_running or (
                    file_being_previewed != selected_file_for_preview
                )
                if st.button(
                    '⏹️ Parar Visualização',
                    disabled=stop_disabled,
                    use_container_width=True,
                ):
                    stop_preview()   # Will stop and rerun

    st.divider()

    # --- Preview Display ---
    st.subheader('Janela de Visualização')
    if is_preview_running:
        # Check if the running preview matches the file selected in the workspace
        if file_being_previewed == selected_file_for_preview:
            st.info(f'Mostrando visualização para `{file_being_previewed}`')
            st.caption(f'URL: {preview_url}')
            # Check if the process is still alive before showing iframe
            live_process = st.session_state.preview_process
            if live_process and live_process.poll() is None:
                # Display the running Streamlit app in an iframe
                st.components.v1.iframe(
                    preview_url, height=600, scrolling=True
                )
            else:
                # The process died unexpectedly
                st.warning(
                    f'Visualização para `{file_being_previewed}` parou inesperadamente.'
                )
                # Attempt to show error output if available
                if live_process:
                    try:
                        stderr = live_process.stderr.read()
                        if stderr:
                            with st.expander(
                                'Mostrar saída de erro do processo parado'
                            ):
                                st.code(stderr)
                    except Exception:
                        pass   # Ignore errors reading output
                # Clear the dead process state (stop_preview handles this and reruns)
                if live_process:   # Check again in case state changed
                    stop_preview()
        else:
            # A preview is running, but not for the file selected in the workspace
            st.warning(
                f'Visualização em execução para `{file_being_previewed}`. Selecione esse arquivo no Workspace para vê-lo aqui, ou pare-o usando os controles acima.'
            )
    else:
        # No preview is currently running
        st.info(
            "Clique 'Executar Visualização' em um arquivo Python selecionado para vê-lo aqui."
        )
