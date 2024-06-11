# ai574 - NLP Financial Chatbot

## Getting Started

### Prereqs

1. Install Git Bash: <https://git-scm.com/downloads>
1. Install VS Code: <https://code.visualstudio.com/>
1. Make sure you have python 3.11 (or above) installed (<https://www.python.org/downloads/>)

### SSH Setup for Github

1. If you're already able to pull Github projects with ssh keys, skip this section
1. Open Git Bash
1. Execute:

    ```bash
    ssh-keygen -o
    ```

    * Take the default options
    * Copy the location of the key (i.e., /c/Users/clayt/.ssh/id_rsa)
1. Execute:

    ```bash
    cat <paste the path from the previous step>
    ```

1. Copy the output. It should look something like:

    ```bash
    ssh-rsa AAAAB3NzaC1yc2EAAAABIwAAAQEAklOUpkDHrfHY17SbrmTIpNLTGK9Tjom/BWDSU
    GPl+nafzlHDTYW7hdI4yZ5ew18JH4JW9jbhUFrviQzM7xlELEVf4h9lFX5QVkbPppSwg0cda3
    Pbv7kOdJ/MTyBlWXFCR+HAo3FXRitBqxiX1nKhXpHAZsMciLq8V6RjsNAQwdsdMFvSlVK/7XA
    t3FaoJoAsncM1Q9x5+3V0Ww68/eIFmb1zuUFljQJKprrX88XypNDvjYNby6vw/Pb0rwert/En
    mZ+AW4OZPnTPI89ZPmVMLuayrD2cE86Z/il8b+gw3r3+1nKatmIkjn2so1d01QraTlMqVSsbx
    NrRFi9wrf+M7Q== schacon@mylaptop.local
    ```

1. Login to Github (<https://github.com/>) in a browser
1. Click your avatar in the upper right corner
1. Click __Settings__
1. Click __SSH and GPG keys__ on the left navigation
1. Click __New SSH key__
1. Paste the copied key from the previous step in the key field
1. Press __Add SSH key__

### Configure VS Code

1. Open VS Code
1. Press __ctrl+shift+x__ to open the extensions pane
1. Search for __Python__
1. Click __Install__ on the one that is published by Microsoft (its probably the top one)
1. Search for __Pylint__
1. Click __Install__ on the one that is published by Microsoft (its probably the top one)
1. Press __ctrl+,__ to bring up settings
1. Search for ___Default Profile__
1. For __Terminal > Integrated > Default Profile: Windows__ select __Git Bash__
1. Close and open __VS Code__
1. Press __Ctrl+Shift+`__ to open a new terminal
1. Navigate to where you want to store your project
1. Execute:

    ```bash
    git clone git@github.com:ClaytonSnyder/ai574.git
    cd Pneumonia-DNN
    code .
    ```

1. A new instance of VS Code will open from the root of your repo directory

### Configure your workspace

1. In VS Code terminal in the root of your repo execute:

    ```bash
    pip3 install poetry
    poetry install
    ```

1. Press __Ctrl+Shift+p__ to open the command palette
1. Type ___Python:___
1. Click __Python: Select Interpreter__
1. Select the one that ends in __('.venv': Poetry)
1. You're all configured
