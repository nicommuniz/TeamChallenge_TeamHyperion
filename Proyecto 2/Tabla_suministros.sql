CREATE TABLE Suministros (
    CodigoProveedor INT,
    CodigoPieza INT,
    Fecha DATE,
    Cantidad INT,
    PRIMARY KEY (CodigoProveedor, CodigoPieza, Fecha),
    FOREIGN KEY (CodigoProveedor) REFERENCES Proveedores(CodigoProveedor),
    FOREIGN KEY (CodigoPieza) REFERENCES Piezas(CodigoPieza)
);
