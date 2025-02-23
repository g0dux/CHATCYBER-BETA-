Below is an interactive README in English that explains how to install, run, and use the Cyber Assistant v4.0 tool, including recommendations for using a virtual environment locally or deploying it on a server.

```markdown
# Cyber Assistant v4.0

Cyber Assistant v4.0 is an integrated cyber investigation and chat tool built with Flask. It provides functionalities for interactive chatting, forensic investigation (with a detailed report and table of found links), and image metadata analysis—all through a responsive web interface that works on desktops and mobile devices.

## Features

- **Chat Mode:** Engage in interactive conversations with an AI model.
- **Investigation Mode:** Perform online investigations with forensic analysis, including:
  - Customizable "Meta of sites" (number of search results)
  - Optional activation of "News" and "Leaked Data" searches
  - An additional "Focus" field for targeted inquiries
  - A detailed report along with a table of found links
- **Metadata Mode:** Analyze image metadata (EXIF) from a provided URL.
- **Responsive Design:** Works well on both desktop and mobile devices.
- **Loading Spinner:** Visual feedback while processing requests.

## Prerequisites

- Python 3.7 or later
- It's **highly recommended** to use a [virtual environment](https://docs.python.org/3/tutorial/venv.html) for local installations.
- Basic familiarity with the command line

## Installation

### 1. Clone the Repository

```bash
git clone https://github.com/yourusername/cyber-assistant-v4.git
cd cyber-assistant-v4
```

### 2. Create a Virtual Environment (Recommended for Local Use)

#### On Linux/Mac:
```bash
python3 -m venv venv
source venv/bin/activate
```

#### On Windows:
```bash
python -m venv venv
venv\Scripts\activate
```

### 3. Install Dependencies

Install the required Python packages using pip:

```bash
pip install -r requirements.txt
```

Make sure the following packages are installed (among others):
- Flask
- nltk
- requests
- psutil
- cachetools
- emoji
- llama_cpp
- huggingface_hub
- duckduckgo_search
- Pillow
- langdetect

*Note: Some packages (e.g., `llama_cpp`) might require additional setup or specific hardware configurations.*

## Running the Tool

### Locally (Development)

To run the tool on your local machine:

```bash
python app.py
```

By default, the Flask server will start on `http://0.0.0.0:5000`. You can access it on your computer's browser. To use it on your mobile device (while on the same network), find your computer's local IP (using `ipconfig` on Windows or `ifconfig` on Linux/Mac) and visit:

```
http://<your-computer-ip>:5000
```

### On a Server

If you plan to deploy the tool on a server:

1. **Allow External Connections:**  
   Ensure the Flask app is running with `app.run(host='0.0.0.0', port=5000, debug=True)` so that it listens on all network interfaces.

2. **Firewall Settings:**  
   Make sure your server’s firewall permits traffic on the designated port (default: 5000).

3. **Production Deployment:**  
   Consider using a production-ready server (e.g., Gunicorn or uWSGI behind Nginx).

Example using Gunicorn:
```bash
gunicorn -w 4 -b 0.0.0.0:5000 app:app
```

## Usage

When you access the tool via your browser:

1. **Select the Mode:**  
   - **Chat:** For general AI conversations.
   - **Investigation:** For forensic investigation. When selected, extra options (Meta of sites, Focus, Activate News, Activate Leaked Data) appear.
   - **Metadados (Metadata):** For image metadata analysis. Input the image URL and get the metadata in the main chat window.

2. **Enter Your Query/URL:**  
   - For **Investigation**, type your target subject.
   - For **Metadata**, type or paste an image URL.
   - For **Chat**, simply enter your message.

3. **Loading Spinner:**  
   A loading spinner will be shown while the server processes your request.

4. **View Results:**  
   The response (which may include formatted HTML such as tables for investigation mode) will appear in the main chat window.

## Contributing

Contributions are welcome! Please open issues or submit pull requests for improvements, bug fixes, or new features.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for more details.

## Contact

For questions, feedback, or support, please open an issue on GitHub or contact [your-email@example.com].

Enjoy using Cyber Assistant v4.0!
```

Simply save this content as `README.md` in your GitHub repository. It provides clear, step-by-step instructions and details on how to run the tool locally or on a server, as well as information about the different modes and features.
