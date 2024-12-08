import mysql.connector

# Establecer la conexión
connection = mysql.connector.connect(
    host="localhost",      # Dirección del servidor (localhost si estás trabajando localmente)
    user="root",           # Usuario de la base de datos (cambia 'root' por el tuyo si es diferente)
    password="tu_contraseña",  # La contraseña de MySQL
    database="proveedores_db"  # El nombre de tu base de datos
)

if connection.is_connected():
    print("Conexión exitosa")
else:
    print("Error en la conexión")

cursor = connection.cursor()
cursor.execute("SELECT * FROM Proveedores")
proveedores = cursor.fetchall()

for proveedor in proveedores:
    print(proveedor)

cursor.close()
connection.close()
