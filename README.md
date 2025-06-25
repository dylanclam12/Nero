# Nero - Options Analysis Tool

A web-based options analysis tool that uses OCR (Optical Character Recognition) to extract data from options chain images and perform risk analysis calculations.

## 🌐 Live Demo

Visit the live application: [Nero Options Tool](https://share.streamlit.io/yourusername/nero)

## 🚀 Features

- **OCR Image Processing**: Upload options chain screenshots and automatically extract data
- **Real-time Analysis**: Calculate max loss, return on risk, and confidence buffers
- **Interactive Interface**: User-friendly web interface built with Streamlit
- **Data Export**: Download extracted data as CSV files
- **Risk Management**: Built-in buffer calculations for 95.5% confidence intervals

## 📊 What It Extracts

From options chain images, the tool extracts:
- Strike prices
- Bid/Ask prices
- Last traded prices
- Delta values
- Implied Volatility (IV)
- Trading volumes
- Buffer calculations for risk management

## 🛠️ Installation & Local Development

### Prerequisites
- Python 3.11 or higher
- PDM (Python Dependency Manager) or pip

### Using PDM (Recommended)
```bash
git clone https://github.com/yourusername/nero.git
cd nero
pdm install
pdm run streamlit run app.py
```

### Using pip
```bash
git clone https://github.com/yourusername/nero.git
cd nero
pip install -r requirements.txt
streamlit run app.py
```

The application will open in your browser at `http://localhost:8501`

## 🌐 Deploy to Streamlit Cloud

1. **Fork this repository** to your GitHub account

2. **Sign up at [Streamlit Cloud](https://share.streamlit.io/)**

3. **Connect your GitHub account** and select this repository

4. **Set the main file path** to `app.py`

5. **Deploy!** Your app will be live at `https://share.streamlit.io/yourusername/nero`

## 🎯 How to Use

1. **Upload an Image**: Click "Choose an options chain image..." and select a screenshot of an options chain
2. **Set Stock Price**: Enter the current stock price in the sidebar for buffer calculations
3. **View Results**: The extracted data will appear in a clean table format
4. **Analyze Options**: Select specific strikes to calculate max loss and return on risk
5. **Download Data**: Export the processed data as a CSV file

## 📁 Project Structure

```
Nero/
├── app.py                 # Main Streamlit application
├── requirements.txt       # Python dependencies
├── pyproject.toml        # PDM configuration
├── .streamlit/           # Streamlit configuration
│   └── config.toml
├── .github/workflows/    # GitHub Actions
│   └── streamlit-app.yml
├── src/
│   ├── data/
│   │   └── test.png     # Sample options chain image
│   └── test.ipynb       # Original Jupyter notebook
└── README.md
```

## 🔧 Configuration

The app includes several configuration options in `.streamlit/config.toml`:
- Custom theme colors
- Upload size limits (10MB)
- Security settings

## 📸 Screenshots

### Main Interface
Upload an options chain image and see the extracted data instantly.

### Analysis Results
Get detailed metrics including max loss, return on risk, and confidence buffers.

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## ⚠️ Disclaimer

This tool is for educational and informational purposes only. It should not be considered as financial advice. Always consult with a qualified financial advisor before making investment decisions.

## 🐛 Issues & Support

If you encounter any issues or have questions:
1. Check the [Issues](https://github.com/yourusername/nero/issues) page
2. Create a new issue if your problem isn't already reported
3. Provide as much detail as possible, including screenshots of any errors

## 📈 Future Enhancements

- [ ] Support for multiple options chain formats
- [ ] Historical data analysis
- [ ] Advanced charting capabilities
- [ ] API integration for real-time data
- [ ] Mobile-responsive design improvements