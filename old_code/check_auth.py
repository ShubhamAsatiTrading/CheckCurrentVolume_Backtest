# check_auth.py - Kite Connect Authentication Troubleshooter

import os
from dotenv import load_dotenv

def check_credentials():
    """Check and validate Kite Connect credentials"""
    print("🔐 KITE CONNECT AUTHENTICATION TROUBLESHOOTER")
    print("=" * 60)
    
    # Load environment
    load_dotenv()
    
    api_key = os.getenv('KITE_API_KEY', '')
    api_secret = os.getenv('KITE_API_SECRET', '')
    
    print(f"📋 Current API Key: {api_key}")
    print(f"📋 Current API Secret: {api_secret[:10]}..." if api_secret else "❌ Not found")
    
    # Check token file
    token = ""
    if os.path.exists('kite_token.txt'):
        with open('kite_token.txt', 'r') as f:
            token = f.read().strip()
        print(f"📋 Current Token: {token[:15]}..." if token else "❌ Empty")
    else:
        print("❌ No token file found")
    
    print("\n" + "-" * 60)
    print("🔧 TROUBLESHOOTING STEPS:")
    print("-" * 60)
    
    # Step 1: Verify API credentials
    print("\n1️⃣ VERIFY API CREDENTIALS:")
    print("   • Login to: https://kite.trade/")
    print("   • Go to: Apps → Create new app (if needed)")
    print("   • Check your API key matches:", api_key)
    print("   • Ensure API secret is correct")
    
    # Step 2: Get fresh token
    print("\n2️⃣ GET FRESH ACCESS TOKEN:")
    print(f"   • Visit: https://kite.zerodha.com/connect/login?api_key={api_key}&v=3")
    print("   • Login with Zerodha credentials")
    print("   • Copy the request_token from URL after login")
    print("   • Paste it in the dashboard when prompted")
    
    # Step 3: Common issues
    print("\n3️⃣ COMMON ISSUES:")
    print("   ❌ Token expired (daily) → Get new token")
    print("   ❌ Wrong API key → Check kite.trade apps section")
    print("   ❌ Wrong API secret → Regenerate secret")
    print("   ❌ App not activated → Activate in kite.trade")
    
    # Step 4: Test connection
    print("\n4️⃣ TEST CONNECTION:")
    if api_key and api_secret and token:
        try:
            from kiteconnect import KiteConnect
            
            print("   🔄 Testing connection...")
            kite = KiteConnect(api_key=api_key)
            kite.set_access_token(token)
            
            # Test with a simple API call
            profile = kite.profile()
            print(f"   ✅ SUCCESS! Connected as: {profile.get('user_name', 'User')}")
            print(f"   ✅ User ID: {profile.get('user_id', 'N/A')}")
            print(f"   ✅ Broker: {profile.get('broker', 'N/A')}")
            
            return True
            
        except Exception as e:
            error_msg = str(e)
            print(f"   ❌ FAILED: {error_msg}")
            
            if "Incorrect `api_key`" in error_msg:
                print("   💡 FIX: Check your API key in .env file")
            elif "access_token" in error_msg:
                print("   💡 FIX: Get a fresh access token (tokens expire daily)")
            elif "API key" in error_msg:
                print("   💡 FIX: Verify API key and secret in kite.trade apps")
            
            return False
    else:
        print("   ❌ Missing credentials - cannot test")
        return False

def interactive_fix():
    """Interactive credential fixing"""
    print("\n" + "=" * 60)
    print("🛠️ INTERACTIVE FIX")
    print("=" * 60)
    
    choice = input("\nWould you like to update credentials? (y/n): ").lower()
    
    if choice == 'y':
        print("\n📝 Enter new credentials (press Enter to skip):")
        
        new_api_key = input("API Key: ").strip()
        new_api_secret = input("API Secret: ").strip()
        new_token = input("Access Token: ").strip()
        
        # Update .env if provided
        if new_api_key or new_api_secret:
            try:
                env_lines = []
                
                if os.path.exists('.env'):
                    with open('.env', 'r') as f:
                        env_lines = f.readlines()
                
                # Update or add lines
                api_key_updated = False
                api_secret_updated = False
                
                for i, line in enumerate(env_lines):
                    if line.startswith('KITE_API_KEY=') and new_api_key:
                        env_lines[i] = f'KITE_API_KEY={new_api_key}\n'
                        api_key_updated = True
                    elif line.startswith('KITE_API_SECRET=') and new_api_secret:
                        env_lines[i] = f'KITE_API_SECRET={new_api_secret}\n'
                        api_secret_updated = True
                
                # Add new lines if not found
                if new_api_key and not api_key_updated:
                    env_lines.append(f'KITE_API_KEY={new_api_key}\n')
                if new_api_secret and not api_secret_updated:
                    env_lines.append(f'KITE_API_SECRET={new_api_secret}\n')
                
                # Write back to .env
                with open('.env', 'w') as f:
                    f.writelines(env_lines)
                
                print("✅ Updated .env file")
                
            except Exception as e:
                print(f"❌ Error updating .env: {e}")
        
        # Update token file if provided
        if new_token:
            try:
                with open('kite_token.txt', 'w') as f:
                    f.write(new_token)
                print("✅ Updated token file")
            except Exception as e:
                print(f"❌ Error updating token: {e}")
        
        # Test again
        if new_api_key or new_api_secret or new_token:
            print("\n🔄 Testing with new credentials...")
            return check_credentials()
    
    return False

if __name__ == "__main__":
    success = check_credentials()
    
    if not success:
        interactive_fix()
    
    print("\n" + "=" * 60)
    print("🎯 NEXT STEPS:")
    print("=" * 60)
    
    if success:
        print("✅ Authentication working! You can now:")
        print("   1. Run: python run.py")
        print("   2. Use the volume analysis system")
    else:
        print("❌ Authentication still failing. Try:")
        print("   1. Get fresh token from kite.zerodha.com")
        print("   2. Verify API key/secret in kite.trade")
        print("   3. Ensure app is activated")
        print("   4. Run this script again")
    
    print("=" * 60)