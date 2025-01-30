# Primera Funcion 

import pandas as pd

df = pd.read_csv("./data/auto_prices.csv")
def describe_df(df):
    
    """
    Analiza un DataFrame y devuelve un resumen con las siguientes filas:
    Columnas: Las trae del dataframe original.
    Filas: 
    - Tipo: El tipo de dato de cada columna.
    - % Nulos: Porcentaje de valores nulos.
    - Valores Únicos: Número de valores únicos.
    - % Cardinalidad: (Valores únicos / total de filas) * 100.
    
    Args:
        df (pd.DataFrame): DataFrame de entrada.
        
    Returns:
        pd.DataFrame: DataFrame con las estadísticas por columna.
    """
    # Crear un diccionario con las métricas requeridas
    filas = {
        'Tipo': df.dtypes.astype(str),  # Convertimos los tipos a string para uniformidad
        '% Nulos': (df.isnull().mean() * 100).round(2),  # Porcentaje de valores nulos
        'Valores Únicos': df.nunique(),  # Número de valores únicos
        '% Cardinalidad': ((df.nunique() / len(df)) * 100).round(2)  # % Cardinalidad
    }
    
    # Convertir el diccionario a un DataFrame transpuesto
    result_df = pd.DataFrame(filas).T
    
    return result_df

describe_df(df)
# Segunda Funcion 

import pandas as pd

def tipifica_variables(dataframe, umbral_categoria, umbral_continua):
    """
    Función para sugerir el tipo de variable basado en la cardinalidad y los umbrales proporcionados.

    Args:
        dataframe (pd.DataFrame): El DataFrame de entrada.
        umbral_categoria (int): Umbral para considerar una variable como categórica.
        umbral_continua (float): Porcentaje mínimo para considerar una variable como numérica continua.

    Returns:
        pd.DataFrame: DataFrame con las columnas "nombre_variable" y "tipo_sugerido".
    """
    resultados = []

    for columna in dataframe.columns:
        cardinalidad = dataframe[columna].nunique()
        total_valores = len(dataframe[columna])
        porcentaje_cardinalidad = cardinalidad / total_valores if total_valores > 0 else 0

        if cardinalidad == 2:
            tipo_sugerido = "Binaria"
        elif cardinalidad < umbral_categoria:
            tipo_sugerido = "Categórica"
        else:
            if porcentaje_cardinalidad >= umbral_continua:
                tipo_sugerido = "Numerica Continua"
            else:
                tipo_sugerido = "Numerica Discreta"

        resultados.append({
            "nombre_variable": columna,
            "tipo_sugerido": tipo_sugerido
        })
        
    df_temp = pd.DataFrame(resultados).reset_index(drop=True)


    return df_temp


tipifica_variables(df, 10,30)
# Funcion 3

def get_features_num_regression(df, target_col, umbral_corr, pvalue=None):
    """
    Selecciona columnas numéricas basadas en su correlación con una variable target.

    Argumentos:
    df (pd.DataFrame): DataFrame de entrada
    target_col (str): Nombre de la columna objetivo (debe ser numérica continua o discreta con alta cardinalidad)
    umbral_corr (float): Umbral de correlación (valor absoluto) entre 0 y 1
    pvalue (floatm opcional): Valor para la prueba de hipótesis, debe estar entre 0 y 1m por defecto es 'None'

    Retorna:
    list: Lista de nombres de columnas numéricas que cumplen los criterios establecidos o None si hay errores en los parámetros de entrada
    
    """
# Verificar que df es un DataFrame
    if not isinstance(df, pd.DataFrame):
        print("Error: 'df' debe ser un DataFrame.")
        return None

    # Verificar que target_col es un string y está en el DataFrame
    if not isinstance(target_col, str) or target_col not in df.columns:
        print(f"Error: 'target_col' debe ser un string existente en el DataFrame.")
        return None

    # Verificar que umbral_corr está en el rango correcto
    if not isinstance(umbral_corr, (int, float)) or not (0 <= umbral_corr <= 1):
        print("Error: 'umbral_corr' debe ser un número entre 0 y 1.")
        return None

    # Verificar que pvalue es correcto si no es None
    if pvalue is not None and (not isinstance(pvalue, float) or not (0 < pvalue < 1)):
        print("Error: 'pvalue' debe ser None o un número entre 0 y 1.")
        return None

    # Verificar que target_col es numérica y tiene alta cardinalidad
    if not np.issubdtype(df[target_col].dtype, np.number):
        print("Error: 'target_col' debe ser numérica.")
        return None

    # Seleccionar columnas numéricas
    numeric_cols = df.select_dtypes(include=[np.number]).columns.drop(target_col)

    # Filtrar columnas por correlación y p-value
    selected_features = [
        col for col in numeric_cols
        if abs(df[col].corr(df[target_col])) > umbral_corr and
        (pvalue is None or pearsonr(df[col].dropna(), df[target_col].dropna())[1] < pvalue)
    ]

    return selected_features
get_features_num_regression(df, "price", 0.5,pvalue=None)
# Funcion 4
def plot_features_num_regression(df, target_col="", columns=[], umbral_corr=0, pvalue=None):
    
    """
    Genera gráficos pairplot de variables numéricas en un DataFrame que cumplen con un umbral de correlación
    respecto a la variable objetivo. Si la lista de columnas es grande, divide los gráficos en grupos de cinco.

    Argumentos:
    df (pd.DataFrame): DataFrame de entrada.
    target_col (str, opcional): Nombre de la columna objetivo. Debe ser una variable numérica continua, (default: "")
    columns (list, opcional): Lista de nombres de columnas a considerar. Si está vacía, se usan todas las numéricas, (default: [])
    umbral_corr (float, opcional): Umbral de correlación absoluta mínimo para seleccionar variables, (default: 0)
    pvalue (float, opcional): Nivel de significación estadística (default: None)

    Retorna:
    list: Lista de columnas que cumplen los criterios de correlación y significación.
    """

    # Verificaciones de los argumentos de entrada
    if not isinstance(df, pd.DataFrame):
        print("Error: 'df' debe ser un DataFrame.")
        return None

    if not isinstance(target_col, str) or target_col not in df.columns or not pd.api.types.is_numeric_dtype(df[target_col]):
        print("Error: 'target_col' debe ser una columna numérica válida del DataFrame.")
        return None

    if not isinstance(umbral_corr, (int, float)) or not (0 <= umbral_corr <= 1):
        print("Error: 'umbral_corr' debe ser un número entre 0 y 1.")
        return None

    if pvalue is not None:
        if not isinstance(pvalue, (int, float)) or not (0 <= pvalue <= 1):
            print("Error: 'pvalue' debe ser un número entre 0 y 1 o None.")
            return None

    # Inicializar columns si es None
    if columns is None:
        columns = []

    # Si columns está vacío, tomar todas las columnas numéricas excepto target_col
    if not columns:
        columns = df.select_dtypes(include=[np.number]).columns.tolist()
        if target_col in columns:
            columns.remove(target_col)

    # Obtener columnas que cumplen el criterio de correlación y p-value
    selected_columns = get_features_num_regression(df, target_col, umbral_corr, pvalue)

    # Si get_features_num_regression devuelve None
    if selected_columns is None:
        print("Error: No se pudieron calcular las correlaciones.")
        return None

    # Filtrar solo las columnas que están en ambas listas
    final_columns = [col for col in columns if col in selected_columns]

    if len(final_columns) == 0:
        print("No hay columnas que cumplan los criterios especificados.")
        return []

    # Dividir en grupos de 5 columnas + target_col
    max_cols_per_plot = 5
    for i in range(0, len(final_columns), max_cols_per_plot - 1):
        subset = [target_col] + final_columns[i:i + (max_cols_per_plot - 1)]
        sns.pairplot(df[subset])
        plt.show()

    return final_columns

plot_features_num_regression(df, target_col="price", columns=[], umbral_corr=0, pvalue=None)

# Funcion 5

def get_features_cat_regression(df, target_col, pvalue=0.05, cardinalidad_max=20):
    """
    Selecciona columnas categóricas relacionadas con la variable objetivo mediante pruebas estadísticas.

    Args:
        df (pd.DataFrame): DataFrame de entrada.
        target_col (str): Nombre de la columna objetivo. Debe ser numérica y continua.
        pvalue (float): Nivel de significación estadística (default: 0.05).
        cardinalidad_max (int): Máximo número de valores únicos para considerar una columna como categórica.

    Returns:
        list: Lista de columnas categóricas relacionadas estadísticamente con la variable objetivo,
              o None si los valores de entrada no son válidos.
    """
    # Verificaciones de entrada
    if not isinstance(df, pd.DataFrame):
        print("Error: 'df' debe ser un DataFrame.")
        return None

    if not isinstance(target_col, str) or target_col not in df.columns or not pd.api.types.is_numeric_dtype(df[target_col]):
        print("Error: 'target_col' debe ser una columna numérica válida del DataFrame.")
        return None

    if not isinstance(pvalue, (int, float)) or not (0 <= pvalue <= 1):
        print("Error: 'pvalue' debe ser un número entre 0 y 1 inclusivo.")
        return None

    # Identificar columnas categóricas (object, category) o numéricas con baja cardinalidad
    potential_categorical_cols = [
        col for col in df.columns if col != target_col and 
        (pd.api.types.is_object_dtype(df[col]) or 
         pd.api.types.is_categorical_dtype(df[col]) or 
         (pd.api.types.is_numeric_dtype(df[col]) and df[col].nunique() <= cardinalidad_max))
    ]

    selected_features = []

    for col in potential_categorical_cols:
        # Crear tabla de contingencia
        contingency_table = pd.crosstab(df[col], df[target_col])

        try:
            if contingency_table.shape[0] > 1:
                # Test ANOVA si la cardinalidad es suficiente, Chi-cuadrado en caso contrario
                if df[col].nunique() > 2:
                    f_stat, p_val = f_oneway(*[df[target_col][df[col] == level].dropna() for level in df[col].unique()])
                else:
                    _, p_val, _, _ = chi2_contingency(contingency_table)

                if p_val <= pvalue:
                    selected_features.append(col)
        except Exception as e:
            print(f"Error procesando la columna {col}: {e}")

    return selected_features


selected_features = get_features_cat_regression(df, "price", pvalue=0.05, cardinalidad_max=10)
selected_features
# Funcion 6

def plot_features_cat_regression(df, target_col="", columns=[], pvalue=0.05, cardinalidad_max=20, with_individual_plot=False):
    """
    Genera histogramas agrupados de una variable objetivo frente a columnas categóricas relacionadas.

    Args:
        df (pd.DataFrame): DataFrame de entrada.
        target_col (str): Nombre de la columna objetivo. Debe ser numérica.
        columns (list): Lista de columnas categóricas a considerar.
        pvalue (float): Nivel de significación estadística.
        cardinalidad_max (int): Máximo número de valores únicos para considerar una columna como categórica.
        with_individual_plot (bool): Si True, genera gráficos individuales para cada categoría.

    Returns:
        list: Lista de columnas categóricas que cumplen con el nivel de significación estadística.
    """


    if not isinstance(df, pd.DataFrame):
        print("Error: 'df' debe ser un DataFrame.")
        return None

    if not isinstance(target_col, str) or target_col not in df.columns or not pd.api.types.is_numeric_dtype(df[target_col]):
        print("Error: 'target_col' debe ser una columna numérica válida del DataFrame.")
        return None

    if not isinstance(columns, list):
        print("Error: 'columns' debe ser una lista de nombres de columnas.")
        return None

    if not isinstance(pvalue, (int, float)) or not (0 <= pvalue <= 1):
        print("Error: 'pvalue' debe ser un número entre 0 y 1 inclusivo.")
        return None

    # Si columns está vacío, identificar posibles categóricas
    if not columns:
        columns = [
            col for col in df.columns if col != target_col and (
                pd.api.types.is_object_dtype(df[col]) or
                pd.api.types.is_categorical_dtype(df[col]) or
                (pd.api.types.is_numeric_dtype(df[col]) and df[col].nunique() <= cardinalidad_max)
            )
        ]

    significant_columns = []

    for col in columns:
        # Crear tabla de contingencia
        contingency_table = pd.crosstab(df[col], df[target_col])

        try:
            if contingency_table.shape[0] > 1:
                # Test ANOVA si la cardinalidad es suficiente, Chi-cuadrado en caso contrario
                if df[col].nunique() > 2:
                    f_stat, p_val = f_oneway(*[df[target_col][df[col] == level].dropna() for level in df[col].unique()])
                else:
                    _, p_val, _, _ = chi2_contingency(contingency_table)

                if p_val < pvalue:
                    significant_columns.append(col)

                    # Generar gráficos
                    if with_individual_plot:
                        plt.figure(figsize=(8, 6))
                        sns.boxplot(x=col, y=target_col, data=df)
                        plt.title(f"Boxplot de {target_col} por {col}")
                        plt.xticks(rotation=45)
                        plt.show()

        except Exception as e:
            print(f"Error procesando la columna {col}: {e}")

    return significant_columns

significant_features = plot_features_cat_regression(
    df, 
    target_col="price", 
    columns=[], 
    pvalue=0.05, 
    cardinalidad_max=10, 
    with_individual_plot=True  # Activar generación de gráficos
)
