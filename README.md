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

1. Unzip the exported WhatsApp chat file.
2. Initialise the `WhatsAppExtractor` class, and pass the path to the chat file `_chat.txt` as an argument to the `extract_chat_data` method to extract the chat messages.
   ```python
   from WhatsAppExtractor import WhatsAppExtractor
   
   extractor = WhatsAppExtractor()
   chat_data = extractor.extract_chat_data('path/to/chat/file/_chat.txt')
   ```