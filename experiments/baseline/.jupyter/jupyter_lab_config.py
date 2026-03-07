# Configuração do Jupyter Lab para IoT-IDS Research
c = get_config()

# Configurações de segurança (apenas para desenvolvimento)
c.ServerApp.token = ''
c.ServerApp.password = ''
c.ServerApp.disable_check_xsrf = True

# Configurações de rede
c.ServerApp.ip = '0.0.0.0'
c.ServerApp.port = 8888
c.ServerApp.open_browser = False
c.ServerApp.allow_root = True

# Diretórios
c.ServerApp.notebook_dir = '/workspace'
c.ServerApp.root_dir = '/workspace'

# Configurações de runtime (usar /tmp para evitar problemas de permissão)
c.ServerApp.runtime_dir = '/tmp/jupyter_runtime'
c.ServerApp.data_dir = '/workspace/.local/share/jupyter'
c.ServerApp.config_dir = '/workspace/.jupyter'

# Desabilitar escrita de arquivos problemáticos
c.ServerApp.browser_open_file = ''

# Configurações adicionais para estabilidade
c.ServerApp.notebook_dir = '/workspace'
c.ServerApp.preferred_dir = '/workspace'