# ai574 - NLP Financial Chatbot

## Getting Started - Reproduction

### Prereqs

1. Make sure you have python 3.11 (or above) installed (<https://www.python.org/downloads/>)
2. Extract the source code zip
3. Install the dependencies:

    ```bash
    pip3 install -r requirements.txt
    python -m spacy download en_core_web_sm
    ```
4. Include the source code folder in your PYTHONPATH:
   1. For windows:
        ```bash
        set PYTHONPATH=%PYTHONPATH%;C:\path\to\extracted\source\
        ```
   2. For Linuz/Mac OS:
        ```bash
        export PYTHONPATH=${PYTHONPATH};/path/to/extracted/source
        ```

### Running the notebook
1. Run the Model 1- GPT2.ipynb notbook
2. Run the Model 2- Langchain-RAG.ipynb notbook

### Running the user interface
1. Install Nodejs (https://nodejs.org/en)
2. Extract the dataset zip into the source code folder (https://pennstateoffice365.sharepoint.com/:u:/s/AI-574/ETb9x_EL_x5Bm-AeP_X-4yIBpwFtKJUN0MP_ksbYQWt87Q?e=y0MFq5)
3. Extract the model zip into the source code folder (https://pennstateoffice365.sharepoint.com/:u:/s/AI-574/Ef4v2vOep55Bvepv0atx0xUB_XLFls95cmohL1WYgdTjMg?e=9o1XMv)
4. From the root of the source code folder, execute:
   ```bash
   python ./fy_bot/server.py --no-debugger --no-reload --host=0.0.0.0
   ```
5. Open another terminal in the root of the source code folder, execute:
   ```bash
   cd fy_bot_ui
   npm install
   npm run dev
   ```
6. Open a browser and go to http://localhost:8080

## Development of Chatbot

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
    cd ai574
    code .
    ```

1. A new instance of VS Code will open from the root of your repo directory

### Configure your workspace

1. In VS Code terminal in the root of your repo execute:

    ```bash
    pip3 install -r requirements.txt
    python -m spacy download en_core_web_sm
    ```


## Running the application
1. If you want to run the notebook example then open the example.ipynb and press play
2. If you want to run the website:
   1. Open the debug menu in VS Code and press the play button (this will start the server)
   2. In a new terminal execute:

      ```bash
      cd fy_bot_ui
      npm run dev
      ```

   3. Click "Yes" when it prompts if it should open a browser
   4. Chat with the bot :)
