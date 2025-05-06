import requests
import bs4
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import TimeoutException, NoSuchElementException
#from webdriver_manager.chrome import ChromeDriverManager
from selenium.webdriver.chrome.service import Service as ChromeService
from webdriver_manager.chrome import ChromeDriverManager
import pandas as pd
import time

table_url = "https://cooperativeelectionstudy.shinyapps.io/ccsearch/"
driver = webdriver.Chrome(service=ChromeService(ChromeDriverManager().install()))
driver.get(table_url)

# Wait for the table to be present and visible (timeout after 10 seconds)
try:
    table_element = WebDriverWait(driver, 10).until(
        EC.presence_of_element_located((By.ID, "DataTables_Table_0"))
    )
    
    # Get headers first
    headers = []
    header_cells = table_element.find_elements(By.TAG_NAME, "th")
    for header in header_cells:
        headers.append(header.text)
    
    # Initialize list to store all data
    all_data = []
    
    while True:
        # Wait for table to load after page change
        time.sleep(2)  # Give the table time to refresh
        
        # Get current page's data
        rows = table_element.find_elements(By.TAG_NAME, "tr")
        
        # Process rows on current page
        for row in rows[1:]:  # Skip header row
            cells = row.find_elements(By.TAG_NAME, "td")
            if cells:
                row_data = [cell.text for cell in cells]
                all_data.append(row_data)
        
        # Try to find and click the "Next" button
        try:
            next_button = WebDriverWait(driver, 5).until(
                EC.presence_of_element_located((By.CSS_SELECTOR, "#DataTables_Table_0_next"))
            )
            
            # Check if "Next" button is disabled
            if 'disabled' in next_button.get_attribute('class'):
                print("Reached last page")
                break
                
            # Click "Next" and wait for table to update
            next_button.click()
            
        except (TimeoutException, NoSuchElementException):
            print("No more pages or couldn't find Next button")
            break
    
    # Convert all collected data to DataFrame
    df = pd.DataFrame(all_data, columns=headers)
    print(f"\nTotal rows collected: {len(df)}")
    print("\nFirst few rows of data:")
    print(df.head())
    
    # Optionally save all data to CSV
    df.to_csv('~/Documents/Scraper_Surveys/ces_shiny_data.csv', index=False)
    #print("\nComplete data saved to complete_table_data.csv")

except Exception as e:
    print(f"Error processing table: {e}")
finally:
    driver.close()

# r = requests.get()
# soup = bs4.BeautifulSoup(r.content, "html.parser")

