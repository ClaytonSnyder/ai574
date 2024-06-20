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

## Running the application

### Build the code/Activate Virtual Environment

1. Ensure that you've done all of the "Getting started" steps
1. Execute:

    ```bash
    poetry build
    poetry install
    source .venv/Scripts/activate
    python -m spacy download en_core_web_sm
    ```

### Create a "project"

1. Execute:

    ```bash
    fybot project create my_project
    ```

    * Notice that a new folder called "projects" was created
    * Notice that a folder called "my_project" was created

### Add datasources to your project

1. Execute:

    ```bash
    fybot datasource download my_project https://www.irs.gov/pub/ebook/p17.epub
    fybot datasource download my_project https://d18rn0p25nwr6d.cloudfront.net/CIK-0001045810/1cbe8fe7-e08a-46e3-8dcc-b429fc06c1a4.pdf
    fybot datasource download my_project https://ir.tesla.com/_flysystem/s3/sec/000162828024002390/tsla-20231231-gen.pdf
    fybot datasource download my_project https://abc.xyz/assets/43/44/675b83d7455885c4615d848d52a4/goog-10-k-2023.pdf
    fybot datasource download my_project https://d18rn0p25nwr6d.cloudfront.net/CIK-0001018724/336d8745-ea82-40a5-9acc-1a89df23d0f3.pdf
    fybot datasource download my_project https://d18rn0p25nwr6d.cloudfront.net/CIK-0001326801/c7318154-f6ae-4866-89fa-f0c589f2ee3d.pdf
    fybot datasource download my_project https://s2.q4cdn.com/299287126/files/doc_financials/2024/q1/Q124-Amazon-Transcript-FINAL.pdf
    fybot datasource download my_project https://ir.tesla.com/_flysystem/s3/sec/000110465924051405/tm2326076d20_defa14a-gen.pdf
    ```

    * Notice that inside the projects/my_project/downloads there is now an p17.epub file
    * Notice that inside the projects/my_project/raw there is now a p17.epub.raw.txt file
