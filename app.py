import streamlit as st
import pandas as pd
import easyocr
import math
import io
from PIL import Image, ImageEnhance, ImageFilter
import numpy as np
import cv2

st.set_page_config(
    page_title="Nero - Options Analysis Tool",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

@st.cache_resource
def load_ocr_reader():
    """Load EasyOCR reader with caching"""
    return easyocr.Reader(['en'])

def preprocess_image(image, method='enhance'):
    """
    Preprocess image to improve OCR accuracy
    """
    # Convert PIL to numpy array
    img_array = np.array(image)
    
    if method == 'enhance':
        # Enhance contrast and sharpness
        enhancer = ImageEnhance.Contrast(image)
        image = enhancer.enhance(1.5)
        
        enhancer = ImageEnhance.Sharpness(image)
        image = enhancer.enhance(1.2)
        
        return np.array(image)
    
    elif method == 'threshold':
        # Convert to grayscale and apply threshold
        gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
        # Apply adaptive threshold
        thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
        return thresh
    
    elif method == 'denoise':
        # Denoise the image
        gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
        denoised = cv2.fastNlMeansDenoising(gray)
        return denoised
    
    elif method == 'morph':
        # Morphological operations to clean up text
        gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
        kernel = np.ones((1,1), np.uint8)
        cleaned = cv2.morphologyEx(gray, cv2.MORPH_CLOSE, kernel)
        return cleaned
    
    return img_array

def extract_table_robust_ocr(image, price=None):
    """
    Enhanced OCR extraction with multiple attempts and preprocessing strategies
    """
    reader = load_ocr_reader()
    
    # Different preprocessing strategies to try
    strategies = [
        ('original', np.array(image)),
        ('enhance', preprocess_image(image, 'enhance')),
        ('threshold', preprocess_image(image, 'threshold')),
        ('denoise', preprocess_image(image, 'denoise')),
        ('morph', preprocess_image(image, 'morph'))
    ]
    
    best_result = None
    best_score = 0
    results_summary = []
    
    for strategy_name, processed_image in strategies:
        try:
            # Try OCR with different confidence thresholds
            for confidence_threshold in [0.3, 0.5, 0.7]:
                results = reader.readtext(processed_image)
                
                # Count valid detections
                valid_detections = [r for r in results if r[2] > confidence_threshold]
                
                if valid_detections:
                    # Score based on number of detections and average confidence
                    avg_confidence = sum(r[2] for r in valid_detections) / len(valid_detections)
                    score = len(valid_detections) * avg_confidence
                    
                    results_summary.append({
                        'strategy': strategy_name,
                        'confidence_threshold': confidence_threshold,
                        'detections': len(valid_detections),
                        'avg_confidence': avg_confidence,
                        'score': score
                    })
                    
                    if score > best_score:
                        best_score = score
                        best_result = valid_detections
        
        except Exception as e:
            st.warning(f"Strategy '{strategy_name}' failed: {str(e)}")
            continue
    
    # Show processing summary in expander
    with st.expander("🔍 OCR Processing Summary", expanded=False):
        if results_summary:
            summary_df = pd.DataFrame(results_summary)
            st.dataframe(summary_df.sort_values('score', ascending=False))
        else:
            st.write("No successful OCR results")
    
    if not best_result:
        st.error("❌ OCR failed with all strategies. Try a higher quality image.")
        return pd.DataFrame()
    
    return process_ocr_results(best_result, price)

def process_ocr_results(results, price=None):
    """
    Process OCR results into a structured DataFrame
    """
    try:
        # Group text by vertical position (rows)
        text_boxes = []
        for (bbox, text, confidence) in results:
            # Get bounding box coordinates
            top_left = bbox[0]
            bottom_right = bbox[2]
            
            text_boxes.append({
                'text': text,
                'x': top_left[0],
                'y': top_left[1],
                'width': bottom_right[0] - top_left[0],
                'height': bottom_right[1] - top_left[1],
                'confidence': confidence
            })
        
        if not text_boxes:
            return pd.DataFrame()
        
        # Sort by Y coordinate (top to bottom)
        text_boxes.sort(key=lambda x: x['y'])
        
        # Group text boxes into rows based on Y coordinate with adaptive tolerance
        rows = []
        current_row = []
        current_y = text_boxes[0]['y']
        
        # Calculate adaptive tolerance based on image size
        avg_height = sum(box['height'] for box in text_boxes) / len(text_boxes)
        tolerance = max(15, avg_height * 0.5)
        
        for box in text_boxes:
            if abs(box['y'] - current_y) <= tolerance:
                current_row.append(box)
            else:
                if current_row:
                    # Sort current row by X coordinate (left to right)
                    current_row.sort(key=lambda x: x['x'])
                    row_text = [box['text'] for box in current_row]
                    rows.append(row_text)
                current_row = [box]
                current_y = box['y']
        
        # Add the last row
        if current_row:
            current_row.sort(key=lambda x: x['x'])
            row_text = [box['text'] for box in current_row]
            rows.append(row_text)
        
        if not rows:
            return pd.DataFrame()
        
        # Filter out rows that are likely headers or irrelevant
        filtered_rows = []
        for row in rows:
            # Skip rows with too few columns or non-numeric data
            if len(row) >= 4:
                # Check if row contains numeric data (strike prices, etc.)
                numeric_count = 0
                for cell in row:
                    try:
                        # Try to convert to float after cleaning
                        cleaned = str(cell).replace('%', '').replace(',', '').replace('$', '')
                        float(cleaned)
                        numeric_count += 1
                    except:
                        pass
                
                # Keep rows with at least 3 numeric values
                if numeric_count >= 3:
                    filtered_rows.append(row)
        
        if not filtered_rows:
            return pd.DataFrame()
        
        # Make all rows the same length
        max_cols = max(len(row) for row in filtered_rows)
        
        # Pad rows to have the same number of columns
        for row in filtered_rows:
            row.extend([''] * (max_cols - len(row)))
        
        # Create DataFrame with flexible column handling
        if max_cols >= 7:
            headers = ["Strike", "Bid", "Ask", "Last", "Delta", "IV", "Volume"][:max_cols]
        else:
            headers = [f"Col_{i+1}" for i in range(max_cols)]
        
        df = pd.DataFrame(filtered_rows, columns=headers)
        
        # Clean and convert columns to numeric
        for col in df.columns:
            # Clean the data first (remove % signs, etc.)
            df[col] = df[col].astype(str).str.replace('%', '').str.replace(',', '').str.replace('$', '')
            # Convert to float, replacing any unconvertible values with NaN
            df[col] = pd.to_numeric(df[col], errors='coerce')
        
        # Remove rows where Strike is NaN (invalid rows)
        df = df.dropna(subset=['Strike'])
        
        if df.empty:
            return pd.DataFrame()
        
        # Set Strike as index
        df = df.set_index('Strike')
        
        # Calculate Buffer column if price is provided and IV column exists
        if price is not None and 'IV' in df.columns:
            try:
                E = (df['IV'] / math.sqrt(52)) * 0.017
                D = price * (1 - E)
                df['Buffer (95.5% Confidence)'] = round(D - df.index, 1)
            except Exception as e:
                st.warning(f"Could not calculate buffer: {e}")

        return df
    
    except Exception as e:
        st.error(f"Error processing OCR results: {str(e)}")
        return pd.DataFrame()

# Keep the original function as a fallback
def extract_table_easyocr(image, price=None):
    """Original OCR function - used as fallback"""
    try:
        reader = load_ocr_reader()
        image_array = np.array(image)
        results = reader.readtext(image_array)
        return process_ocr_results(results, price)
    except Exception as e:
        st.error(f"Error processing image: {str(e)}")
        return pd.DataFrame()

def calculate_option_metrics(strike_price, premium):
    """
    Calculate Max Loss and Return on Risk Percentage for a specific option trade
    
    Args:
        strike_price: The strike price of the option
        premium: The premium paid/received for the option
    
    Returns:
        max_loss and return_on_risk_percentage
    """
    # Calculate Max Loss = (Strike - premium) * 100
    max_loss = (strike_price - premium) * 100
    
    # Calculate Return on Risk Percentage = (premium / max_loss) * 100
    return_on_risk_percentage = round(((premium * 100) / max_loss) * 100, 2) if max_loss != 0 else 0
    
    return f"${round(max_loss)}", f"{return_on_risk_percentage}%"

def main():
    st.title("📊 Nero - Options Analysis Tool")
    st.markdown("Upload an options chain image to extract and analyze the data")
    
    # Sidebar for inputs
    st.sidebar.header("Configuration")
    
    # File uploader
    uploaded_file = st.file_uploader(
        "Choose an options chain image...",
        type=['png', 'jpg', 'jpeg'],
        help="Upload an image of an options chain table"
    )
    
    # Price input
    stock_price = st.sidebar.number_input(
        "Current Stock Price ($)",
        min_value=0.01,
        value=222.65,
        step=0.01,
        help="Enter the current stock price for buffer calculations"
    )
    
    if uploaded_file is not None:
        # Display the uploaded image
        image = Image.open(uploaded_file)
        
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.subheader("Uploaded Image")
            st.image(image, caption="Options Chain Image", use_container_width=True)
        
        with col2:
            st.subheader("Processing...")
            
            # Process the image with robust OCR
            with st.spinner("Extracting data from image..."):
                # Add OCR method selection
                ocr_method = st.radio(
                    "OCR Method:",
                    ["Robust (Recommended)", "Standard"],
                    help="Robust method tries multiple preprocessing strategies"
                )
                
                if ocr_method == "Robust (Recommended)":
                    df = extract_table_robust_ocr(image, price=stock_price)
                else:
                    df = extract_table_easyocr(image, price=stock_price)
            
            if not df.empty:
                st.success("✅ Data extracted successfully!")
                
                # Display the extracted data
                st.subheader("Extracted Options Data")
                st.dataframe(df, use_container_width=True)
                
                # Options analysis section
                st.subheader("Options Analysis")
                
                # Select strike price for analysis
                available_strikes = df.index.dropna().tolist()
                if available_strikes:
                    selected_strike = st.selectbox(
                        "Select Strike Price for Analysis:",
                        available_strikes
                    )
                    
                    # Get the row data for the selected strike
                    row_data = df.loc[selected_strike]
                    
                    # Display current option data
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        st.metric("Bid", f"${row_data['Bid']:.2f}")
                        st.metric("Ask", f"${row_data['Ask']:.2f}")
                    
                    with col2:
                        st.metric("Last", f"${row_data['Last']:.2f}")
                        st.metric("Delta", f"{row_data['Delta']:.4f}")
                    
                    with col3:
                        st.metric("IV", f"{row_data['IV']:.2f}")
                        st.metric("Volume", f"{int(row_data['Volume'])}")
                    
                    # Buffer calculation if available
                    if 'Buffer (95.5% Confidence)' in df.columns:
                        st.metric("Buffer (95.5% Confidence)", f"${row_data['Buffer (95.5% Confidence)']}")
                    
                    # Option metrics calculation
                    st.subheader("Trade Analysis")
                    
                    premium_input = st.number_input(
                        "Premium (per contract):",
                        min_value=0.01,
                        value=float(row_data['Last']),
                        step=0.01
                    )
                    
                    max_loss, return_on_risk = calculate_option_metrics(selected_strike, premium_input)
                    
                    col1, col2 = st.columns(2)
                    with col1:
                        st.metric("Max Loss", max_loss)
                    with col2:
                        st.metric("Return on Risk", return_on_risk)
                
                # Download processed data
                csv = df.to_csv()
                st.download_button(
                    label="📥 Download CSV",
                    data=csv,
                    file_name="options_data.csv",
                    mime="text/csv"
                )
                
            else:
                st.error("❌ Could not extract data from the image. Please try a different image or check the image quality.")
    
    else:
        st.info("👆 Please upload an options chain image to get started")
        
        # Show example or instructions
        st.subheader("How to use:")
        st.markdown("""
        1. **Upload an image** of an options chain table
        2. **Set the current stock price** in the sidebar
        3. **View the extracted data** in a clean table format
        4. **Analyze specific strikes** to calculate max loss and return on risk
        5. **Download the data** as CSV for further analysis
        
        The tool uses OCR (Optical Character Recognition) to automatically extract:
        - Strike prices
        - Bid/Ask prices
        - Last traded price
        - Delta values
        - Implied Volatility (IV)
        - Trading volume
        - Buffer calculations for risk management
        """)

if __name__ == "__main__":
    main() 