import pandas as pd
from scipy import stats


archivo_especifico = r'C:\Users\Annna\OneDrive\Documentos\Inteligencia artificial\PrediccionCalidadDelAire\24RAMA\2024PM25.xls'


df_pm25 = pd.read_excel(archivo_especifico)

# Reemplazar los valores -99 por NaN
df_pm25.replace(-99, pd.NA, inplace=True)


fecha_col = next((col for col in df_pm25.columns if col.lower() == 'fecha'), None)
if fecha_col:
    df_pm25.rename(columns={fecha_col: 'FECHA'}, inplace=True)
    df_pm25['FECHA'] = pd.to_datetime(df_pm25['FECHA'], unit='ms', errors='coerce')  # Convertir a datetime
else:
    print("Advertencia: No se encontró una columna de fecha en el archivo.")
    exit()

# Dividir los datos por estaciones (verano e invierno) 
summer_data = df_pm25[df_pm25['FECHA'].dt.month.isin([6, 7, 8])]
winter_data = df_pm25[df_pm25['FECHA'].dt.month.isin([12, 1, 2])]

#  "###"
df_pm25['FECHA'] = '###'


columns_to_analyze = df_pm25.select_dtypes(include=['float64', 'int64']).columns.difference(['FECHA', 'HORA'])


df_pm25 = df_pm25.apply(pd.to_numeric, errors='coerce')


numerical_columns = df_pm25.select_dtypes(include=['float64', 'int64']).columns
print("\nColumnas numéricas disponibles para análisis:", numerical_columns)

# almacenar resultados
resultados_estadisticas = []
resultados_normalidad = []
resultados_prueba = []
resultados_ajuste_lognorm = []

# Procesar cada columna por separado
for column_name in numerical_columns:
    if column_name.lower() in ['fecha', 'hora']:
        continue  

    # Verificar si la columna tiene valores válidos
    if df_pm25[column_name].isnull().all():
        continue  # saltamos

    
    if pd.api.types.is_numeric_dtype(df_pm25[column_name]):

        # Extraer los datos de verano e invierno 
        summer_values = summer_data[column_name].dropna().astype(float)
        winter_values = winter_data[column_name].dropna().astype(float)

        # Filtrar los valores negativos o cero antes de ajustar la distribución log-normal
        summer_values_filtered = summer_values[summer_values > 0]
        winter_values_filtered = winter_values[winter_values > 0]

        # Calcular estadísticas descriptivas para cada estación
        summer_stats = summer_values.describe()
        winter_stats = winter_values.describe()

        # Realizar una prueba de normalidad para determinar si usar t de Student o Mann-Whitney U
        normality_summer = stats.shapiro(summer_values)
        normality_winter = stats.shapiro(winter_values)

        # Almacenar los resultados de las pruebas de normalidad
        resultados_normalidad.append({
            'columna': column_name,
            'normalidad_summer': normality_summer,
            'normalidad_winter': normality_winter
        })

        # Si ambos conjuntos de datos son normales, utilizamos la prueba t de Student
        if normality_summer.pvalue > 0.05 and normality_winter.pvalue > 0.05:
            prueba = 't de Student'
            t_stat, p_value = stats.ttest_ind(summer_values, winter_values)
        else:
            prueba = 'Mann-Whitney U'
            u_stat, p_value = stats.mannwhitneyu(summer_values, winter_values)
            t_stat = u_stat

        # Almacenar los resultados de la prueba de diferencia
        resultados_prueba.append({
            'columna': column_name,
            'prueba': prueba,
            'p_value': p_value
        })

        # Ajuste de distribución log-normal 
        params_summer = stats.lognorm.fit(summer_values_filtered)
        _, p_value_lognorm = stats.kstest(summer_values_filtered, 'lognorm', args=params_summer)

        params_winter = stats.lognorm.fit(winter_values_filtered)
        _, p_value_lognorm_winter = stats.kstest(winter_values_filtered, 'lognorm', args=params_winter)

        # Almacenar los resultados del ajuste log-normal
        resultados_ajuste_lognorm.append({
            'columna': column_name,
            'p_value_lognorm_summer': p_value_lognorm,
            'p_value_lognorm_winter': p_value_lognorm_winter
        })

        # Almacenar las estadísticas descriptivas
        resultados_estadisticas.append({
            'columna': column_name,
            'summer_stats': summer_stats,
            'winter_stats': winter_stats
        })

# Imprimir los resultados al final
print("\nEstadísticas descriptivas:")
for resultado in resultados_estadisticas:
    print(f"\nEstadísticas descriptivas para {resultado['columna']} en VERANO:")
    print(resultado['summer_stats'])
    print(f"\nEstadísticas descriptivas para {resultado['columna']} en INVIERNO:")
    print(resultado['winter_stats'])

print("\nResultados de la prueba de normalidad:")
for resultado in resultados_normalidad:
    print(f"\nPara la columna {resultado['columna']}:")
    print(f"Normalidad en VERANO: Estadístico = {resultado['normalidad_summer'].statistic}, p-value = {resultado['normalidad_summer'].pvalue}")
    print(f"Normalidad en INVIERNO: Estadístico = {resultado['normalidad_winter'].statistic}, p-value = {resultado['normalidad_winter'].pvalue}")

print("\nResultados de la prueba de diferencia significativa:")
for resultado in resultados_prueba:
    print(f"\nPara la columna {resultado['columna']}:")
    print(f"Prueba utilizada: {resultado['prueba']}")
    print(f"p-value de la prueba: {resultado['p_value']}")

print("\nResultados del ajuste log-normal:")
for resultado in resultados_ajuste_lognorm:
    print(f"\nPara la columna {resultado['columna']}:")
    print(f"p-value ajuste log-normal en VERANO: {resultado['p_value_lognorm_summer']}")
    print(f"p-value ajuste log-normal en INVIERNO: {resultado['p_value_lognorm_winter']}")

# a) Ordenar el DataFrame por todos los valores numéricos en orden descendente
df_pm25_sorted = df_pm25.sort_values(by=df_pm25.select_dtypes(include=['float64', 'int64']).columns.tolist(), ascending=False)

# b) Seleccionar las primeras 10 filas
top_10_rows = df_pm25_sorted.head(10)

# c) Crear una lista de tuplas 
fecha_valores_tuplas = [(row['FECHA'], row.drop('FECHA').to_dict()) for _, row in top_10_rows.iterrows()]

# Mostrar la lista de tuplas 
print("Las primeras 10 filas ordenadas por valores numéricos en orden descendente son:")
for idx, tupla in enumerate(fecha_valores_tuplas, 1):
    print(f"\nFila {idx}: Fecha: {tupla[0]} | Valores: {tupla[1]}")

# a) Categorización
bins = [0, 12, 35, 55, float('inf')]
labels = ["Bajo", "Moderado", "Alto", "Muy Alto"]

# Categorizar las columnas numéricas
for column in numerical_columns:
    df_pm25[column + '_categoria'] = pd.cut(df_pm25[column], bins=bins, labels=labels, include_lowest=True)

# Guardar el DataFrame con la categorización en un nuevo archivo Excel
df_pm25.to_excel(r'C:\Users\Annna\OneDrive\Documentos\Inteligencia artificial\PrediccionCalidadDelAire\24RAMA\2024PM25_categorizado.xlsx', index=False)
print("\nEl archivo con la categorización ha sido guardado exitosamente como '2024PM25_categorizado.xlsx'.")
