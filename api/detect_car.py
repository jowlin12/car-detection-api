import gspread
from oauth2client.service_account import ServiceAccountCredentials

# Configura las credenciales
scope = ["https://spreadsheets.google.com/feeds", "https://www.googleapis.com/auth/drive"]
creds = ServiceAccountCredentials.from_json_keyfile_name("credentials.json", scope)
client = gspread.authorize(creds)

# Abre la hoja de cálculo por URL
spreadsheet = client.open_by_url("https://docs.google.com/spreadsheets/d/1247WriRrUZSXep9Txj0398oXOtVWLnnI7JO5uS5pCGU/edit")
sheet = spreadsheet.worksheet("Base de Datos")

# Prueba para leer datos
data = sheet.get_all_records()
print(data)
