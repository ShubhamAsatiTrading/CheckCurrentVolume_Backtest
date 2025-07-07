
from kiteconnect import KiteConnect
import os
import json
from datetime import datetime
KITE_API_KEY="gl0qvjm9s81w22cf"
KITE_API_SECRET="l3oqfafm9x96lvl1inv07pikjkbz6gu5"
KITE_REQUEST_TOKEN="WwCI3vGficDREMmoRVTf7TjD1WYS7qIo"
TOKEN_FILE = "kite_token.txt"


def first_run():
    # Check if token exists and is valid for today
    if os.path.exists(TOKEN_FILE):
        try:
            with open(TOKEN_FILE, "r") as f:
                token_data = json.loads(f.read().strip())
                today = datetime.now().strftime("%Y-%m-%d")
                
                if token_data.get("date") == today and token_data.get("access_token"):
                    return token_data["access_token"]
                else:
                    os.remove(TOKEN_FILE)  # Delete old token
        except:
            if os.path.exists(TOKEN_FILE):
                os.remove(TOKEN_FILE)
    
    # Generate new token
    kite = KiteConnect(api_key=KITE_API_KEY)
    login_url = kite.login_url()
    print("\nLogin via this URL:\n", login_url)
    
    request_token = input("Paste request_token: ").strip()
    
    try:
        session = kite.generate_session(request_token, api_secret=KITE_API_SECRET)
        access_token = session["access_token"]
        
        # Save with today's date
        token_data = {"access_token": access_token, "date": datetime.now().strftime("%Y-%m-%d")}
        with open(TOKEN_FILE, "w") as f:
            f.write(json.dumps(token_data))
        
        return access_token
    except Exception as e:
        print("Failed to generate session:", e)
        return None

# Usage Example
if __name__ == "__main__":
    token = first_run()
    if token:
        print("Access Token ready for use in main_code_1.py:", token)
    else:
        print("Could not generate or load access token.")


