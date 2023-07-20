# WhatsApp Analyser

A Python program to extract and analyse exported WhatsApp chat data.

## Installation

1. Ensure you have Python 3.10 or higher installed on your system.
2. Clone the repository.
3. Install the required dependencies using the following command:
   ```
   pip install -r requirements.txt
   ```

## Usage

1. Export the WhatsApp chat data from WhatsApp.
2. Unzip the exported WhatsApp chat data.
3. Initialise the `WhatsAppAnalyser` class with the path to the unzipped directory.
   ```python
   from WhatsAppAnalyser import WhatsAppAnalyser
   
   path_to_chat_data = 'path/to/chat/data'
   analyser = WhatsAppAnalyser(chat_data_path=path_to_chat_data)
   ```

## Example

An example of how to use the `WhatsAppAnalyser` class is shown in the `example.ipynb` file.