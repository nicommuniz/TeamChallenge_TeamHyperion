CREATE DATABASE proveedores_db;
USE proveedores_db;

INSERT INTO Proveedores (CodigoProveedor, Nombre, Direccion, Ciudad, Provincia)
VALUES (1, 'Proveedor Garcia', 'Calle Tramontana 88', 'Mahon', 'Baleares');

INSERT INTO Categorias (CodigoCategoria, Nombre)
VALUES (1, 'Electrónica');

INSERT INTO Piezas (CodigoPieza, Nombre, Color, Precio, CodigoCategoria)
VALUES (101, 'Resistor', 'Rojo', 0.5, 1);

INSERT INTO Suministros (CodigoProveedor, CodigoPieza, Fecha, Cantidad)
VALUES (1, 101, '2024-12-01', 100);

SELECT * FROM Proveedores;
SELECT * FROM Piezas;
SELECT * FROM Suministros;
