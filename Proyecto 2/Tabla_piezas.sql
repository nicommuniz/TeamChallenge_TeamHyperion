CREATE TABLE Piezas (
    CodigoPieza INT PRIMARY KEY,
    Nombre VARCHAR(100),
    Color VARCHAR(50),
    Precio DECIMAL(10,2),
    CodigoCategoria INT,
    FOREIGN KEY (CodigoCategoria) REFERENCES Categorias(CodigoCategoria)
);
