import os
import pandas as pd
import numpy as np
from scipy.integrate import cumulative_trapezoid
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA 
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
import warnings
warnings.filterwarnings("ignore")
from statsmodels.tsa.arima.model import ARIMA
warnings.filterwarnings("ignore")


# Rutas de las carpetas
carpeta_rama = r'C:\Users\Annna\OneDrive\Documentos\Inteligencia artificial\PrediccionCalidadDelAire\24RAMA'
carpeta_redemet = r'C:\Users\Annna\OneDrive\Documentos\Inteligencia artificial\PrediccionCalidadDelAire\24REDMET'

# Función para procesar archivos y calcular matrices de correlación
def procesar_archivos(carpeta):
    matrices_correlacion = []
    derivadas = {}
    integrales = {}  # Almacenar las integrales de cada variable
    estadisticas = {}  # Diccionario para almacenar estadísticas

    # Procesar los archivos en la carpeta proporcionada
    for archivo in os.listdir(carpeta):
        if archivo.endswith('.xls') or archivo.endswith('.xlsx'):
            archivo_path = os.path.join(carpeta, archivo)
            print(f"\nCargando archivo: {archivo_path}...")

            # Para otros archivos, solo cargarlos sin mostrar las columnas
            df = pd.read_excel(archivo_path)

            # Verificar si la columna "FECHA" existe (ignorando mayúsculas/minúsculas)
            fecha_col = next((col for col in df.columns if col.lower() == 'fecha'), None)
            if fecha_col:
                df.rename(columns={fecha_col: 'FECHA'}, inplace=True)
                df['FECHA'] = pd.to_datetime(df['FECHA'], errors='coerce')  # Convertir a datetime
            else:
                print(f"⚠️ Advertencia: No se encontró una columna de fecha en {archivo}")

            # Reemplazar valores -99 por NaN y eliminar columnas completamente vacías
            df.replace(-99, np.nan, inplace=True)
            df.dropna(axis=1, how='all', inplace=True)

            # Interpolación lineal para rellenar valores NaN
            df.interpolate(method='linear', inplace=True)

            # Eliminar las columnas 'FECHA' y 'HORA' de la matriz de correlación
            df.drop(columns=['FECHA', 'HORA'], errors='ignore', inplace=True)

            # Seleccionar solo columnas numéricas
            df_numeric = df.select_dtypes(include=[np.number])

            # Calcular matriz de correlación y almacenar
            correlation_matrix = df_numeric.corr()
            matrices_correlacion.append(correlation_matrix)

            # Calcular la derivada numérica e integral de cada columna
            for col in df_numeric.columns:
                valores = df_numeric[col].dropna().values

                if col not in derivadas:
                    derivadas[col] = []
                derivadas[col].extend(np.gradient(valores))

                if col not in integrales:
                    integrales[col] = []

                # Calcular integral numérica con cumulative_trapezoid
                if len(valores) > 1:
                    tiempo = np.arange(len(valores))  # Suponiendo intervalos regulares
                    integral_col = cumulative_trapezoid(valores, tiempo, initial=0)
                    integrales[col].extend(integral_col)

            # Calcular estadísticas para cada archivo
            for col in df_numeric.columns:
                # Almacenar estadísticas como un sub-diccionario por contaminante
                if col not in estadisticas:
                    estadisticas[col] = {
                        "Media": [],
                        "Mediana": [],
                        "Desviación estándar": [],
                        "Cuartiles": []
                    }

                estadisticas[col]["Media"].append(df_numeric[col].mean())
                estadisticas[col]["Mediana"].append(df_numeric[col].median())
                estadisticas[col]["Desviación estándar"].append(df_numeric[col].std())
                estadisticas[col]["Cuartiles"].append(df_numeric[col].quantile([0.25, 0.5, 0.75]).to_dict())

    return matrices_correlacion, derivadas, integrales, estadisticas

# Procesar archivos en ambas carpetas
matrices_correlacion_rama, derivadas_rama, integrales_rama, estadisticas_rama = procesar_archivos(carpeta_rama)
matrices_correlacion_redemet, derivadas_redemet, integrales_redemet, estadisticas_redemet = procesar_archivos(carpeta_redemet)

# Combinar matrices de correlación
combined_rama = pd.concat(matrices_correlacion_rama, axis=0, ignore_index=True, join='inner') if matrices_correlacion_rama else pd.DataFrame()
combined_redemet = pd.concat(matrices_correlacion_redemet, axis=0, ignore_index=True, join='inner') if matrices_correlacion_redemet else pd.DataFrame()

print("\nMatriz de correlación combinada para RAMA:")
print(combined_rama)

print("\nMatriz de correlación combinada para REDMET:")
print(combined_redemet)

# Verificar si hay matrices de correlación disponibles antes de calcular autovalores/autovectores
if matrices_correlacion_rama:
    avg_correlation_rama = sum(matrices_correlacion_rama) / len(matrices_correlacion_rama)
    avg_correlation_rama.fillna(0, inplace=True)

    eigenvalues_rama, eigenvectors_rama = np.linalg.eigh(avg_correlation_rama)
    print("\nAutovalores RAMA:", eigenvalues_rama)
    print("Autovectores RAMA:", eigenvectors_rama)

if matrices_correlacion_redemet:
    avg_correlation_redemet = sum(matrices_correlacion_redemet) / len(matrices_correlacion_redemet)
    avg_correlation_redemet.fillna(0, inplace=True)

    eigenvalues_redemet, eigenvectors_redemet = np.linalg.eigh(avg_correlation_redemet)
    print("\nAutovalores REDMET:", eigenvalues_redemet)
    print("Autovectores REDMET:", eigenvectors_redemet)

# Derivadas 
print("\nDerivadas calculadas:")
for col, derivada in derivadas_rama.items():
    print(f"{col}: {derivada[:5]}")  

#Integrales
print("\nIntegrales calculadas:")
for col, integral in integrales_rama.items():
    print(f"{col}: {integral[:5]}")  

# Mostrar estadísticas (media, mediana, desviación estándar)
print("\nEstadísticas RAMA:")
for col, stats in estadisticas_rama.items():
    print(f"\n{col}:")
    for stat, values in stats.items():
        print(f"  {stat}: {values}")

print("\nEstadísticas REDMET:")
for col, stats in estadisticas_redemet.items():
    print(f"\n{col}:")
    for stat, values in stats.items():
        print(f"  {stat}: {values}") 

 #PM25 y PM10
ruta_pm25 = r'C:\Users\Annna\OneDrive\Documentos\Inteligencia artificial\PrediccionCalidadDelAire\24RAMA\2024PM25.xls'
ruta_pm10 = r'C:\Users\Annna\OneDrive\Documentos\Inteligencia artificial\PrediccionCalidadDelAire\24RAMA\2024PM10.xls'


df_pm25 = pd.read_excel(ruta_pm25)
df_pm10 = pd.read_excel(ruta_pm10)


df_pm25['FECHA'] = pd.to_datetime(df_pm25['FECHA'], errors='coerce')
df_pm10['FECHA'] = pd.to_datetime(df_pm10['FECHA'], errors='coerce')


sns.set(style="whitegrid")

#  gráfico de series temporales con Seaborn
fig, ax1 = plt.subplots(figsize=(12, 6))

# Grafica PM25 en el primer eje (ax1)
sns.lineplot(data=df_pm25, x='FECHA', y=df_pm25.columns[1], label='PM25', color='blue', linewidth=2, ax=ax1)
ax1.set_xlabel('Fecha', fontsize=12)
ax1.set_ylabel('PM25 (µg/m³)', fontsize=12, color='blue')
ax1.tick_params(axis='y', labelcolor='blue')


ax2 = ax1.twinx()  # Crear un segundo eje que comparte el mismo eje X
sns.lineplot(data=df_pm10, x='FECHA', y=df_pm10.columns[1], label='PM10', color='darkred', linewidth=2, ax=ax2)
ax2.set_ylabel('PM10 (µg/m³)', fontsize=12, color='darkred')
ax2.tick_params(axis='y', labelcolor='darkred')

# Mejorar la visualización de las fechas en el eje X
plt.xticks(rotation=45)


plt.title('Evolución de PM25 y PM10 a lo largo del tiempo', fontsize=14)
ax1.legend(loc='upper left', fontsize=12)
ax2.legend(loc='upper right', fontsize=12)


plt.tight_layout()

plt.show()

# Calcular matrices de correlación específicas para PM25 y PM10
# Asegurarse de que los datos numéricos se usan para las correlaciones
df_pm25_numeric = df_pm25.select_dtypes(include=[np.number])
df_pm10_numeric = df_pm10.select_dtypes(include=[np.number])

# Matriz de correlación PM25
correlation_matrix_pm25 = df_pm25_numeric.corr()

# Matriz de correlación PM10
correlation_matrix_pm10 = df_pm10_numeric.corr()

# Visualización de las matrices de correlación
plt.figure(figsize=(12, 8))

# Heatmap de la matriz de correlación de PM25
plt.subplot(1, 2, 1)
sns.heatmap(correlation_matrix_pm25, annot=True, cmap='coolwarm', fmt='.2f', cbar=True,
            annot_kws={'size': 10}, linewidths=0.5, square=True, vmin=-1, vmax=1)
plt.title('Matriz de Correlación - PM25', fontsize=16)

# Heatmap de la matriz de correlación de PM10
plt.subplot(1, 2, 2)
sns.heatmap(correlation_matrix_pm10, annot=True, cmap='coolwarm', fmt='.2f', cbar=True,
            annot_kws={'size': 10}, linewidths=0.5, square=True, vmin=-1, vmax=1)
plt.title('Matriz de Correlación - PM10', fontsize=16)

# Mejorar el espaciado entre los gráficos
plt.tight_layout()

# Mostrar el gráfico
plt.show()

# Seleccionar las columnas numéricas de los datos
df_rama_numeric = df_pm25.select_dtypes(include=[np.number])  
df_redemet_numeric = df_pm10.select_dtypes(include=[np.number])  

# Estandarizar los datos (importante para PCA)
scaler_rama = StandardScaler()
scaler_redemet = StandardScaler()

# Ajustar y transformar los datos
df_rama_scaled = scaler_rama.fit_transform(df_rama_numeric)
df_redemet_scaled = scaler_redemet.fit_transform(df_redemet_numeric)

# Crear el modelo PCA
pca_rama = PCA(n_components=2) 
pca_redemet = PCA(n_components=2)

# Ajustar PCA y transformar los datos
pca_rama_result = pca_rama.fit_transform(df_rama_scaled)
pca_redemet_result = pca_redemet.fit_transform(df_redemet_scaled)

# Ver la varianza explicada por los componentes principales
print("Varianza explicada por los componentes principales (RAMA):")
print(pca_rama.explained_variance_ratio_)

print("\nVarianza explicada por los componentes principales (REDMET):")
print(pca_redemet.explained_variance_ratio_)

# Ver los componentes principales
print("\nComponentes principales (RAMA):")
print(pca_rama.components_)

print("\nComponentes principales (REDMET):")
print(pca_redemet.components_)

# Graficar los resultados de PCA (si quieres visualizarlos)
plt.figure(figsize=(8, 6))

# Gráfico de dispersión para los primeros dos componentes de RAMA
plt.subplot(1, 2, 1)
plt.scatter(pca_rama_result[:, 0], pca_rama_result[:, 1], c='blue', label='RAMA')
plt.xlabel('Componente 1')
plt.ylabel('Componente 2')
plt.title('PCA RAMA')
plt.legend()

# Gráfico de dispersión para los primeros dos componentes de REDMET
plt.subplot(1, 2, 2)
plt.scatter(pca_redemet_result[:, 0], pca_redemet_result[:, 1], c='red', label='REDMET')
plt.xlabel('Componente 1')
plt.ylabel('Componente 2')
plt.title('PCA REDMET')
plt.legend()

# Mejorar visualización
plt.tight_layout()
plt.show()

# Definir la ruta del archivo
ruta_no2 = r'C:\Users\Annna\OneDrive\Documentos\Inteligencia artificial\PrediccionCalidadDelAire\24RAMA\2024NO2.xls'

# 1. Cargar el archivo
df_no2 = pd.read_excel(ruta_no2)

# 2. Limpiar los datos: convertir -99 en NaN
df_no2.replace(-99, np.nan, inplace=True)

# 3. Crear una columna 'Estacion' según la fecha
df_no2['FECHA'] = pd.to_datetime(df_no2['FECHA'])

def obtener_estacion(fecha):
    mes = fecha.month
    if mes in [12, 1, 2]:
        return 'Invierno'
    elif mes in [3, 4, 5]:
        return 'Primavera'
    elif mes in [6, 7, 8]:
        return 'Verano'
    else:
        return 'Otoño'

df_no2['Estacion'] = df_no2['FECHA'].apply(obtener_estacion)

# 4. Reformatear el dataframe de wide a long
df_long = df_no2.melt(id_vars=['FECHA', 'HORA', 'Estacion'],
                      value_vars=df_no2.columns[2:-1],  # columnas de las estaciones
                      var_name='Sitio',
                      value_name='NO2')

# 5. Crear el boxplot
plt.figure(figsize=(10, 6))
sns.boxplot(x='Estacion', y='NO2', data=df_long, hue='Estacion', palette='Set2', legend=False)


# Configurar el gráfico
plt.title('Distribución de niveles de NO2 por Estación del Año')
plt.xlabel('Estación del Año')
plt.ylabel('Concentración de NO2 (µg/m³)')
plt.grid(True)
plt.show()

# Función para obtener la estación del año 
def obtener_estacion(fecha):
    mes = fecha.month
    if mes in [12, 1, 2]:
        return 'Invierno'
    elif mes in [3, 4, 5]:
        return 'Primavera'
    elif mes in [6, 7, 8]:
        return 'Verano'
    else:
        return 'Otoño'

# 1. Leer los archivos
ruta_pm10 = r'C:\Users\Annna\OneDrive\Documentos\Inteligencia artificial\PrediccionCalidadDelAire\24RAMA\2024PM10.xls'
ruta_no2  = r'C:\Users\Annna\OneDrive\Documentos\Inteligencia artificial\PrediccionCalidadDelAire\24RAMA\2024NO2.xls'
ruta_pm25 = r'C:\Users\Annna\OneDrive\Documentos\Inteligencia artificial\PrediccionCalidadDelAire\24RAMA\2024PM25.xls'

df_pm10 = pd.read_excel(ruta_pm10)
df_no2 = pd.read_excel(ruta_no2)
df_pm25 = pd.read_excel(ruta_pm25)

# 2. Convertir las fechas a tipo datetime
df_pm10['FECHA'] = pd.to_datetime(df_pm10['FECHA'])
df_no2['FECHA'] = pd.to_datetime(df_no2['FECHA'])
df_pm25['FECHA'] = pd.to_datetime(df_pm25['FECHA'])

# 3. Agregar columna 'Estacion' 
df_pm10['Estacion'] = df_pm10['FECHA'].apply(obtener_estacion)
df_no2['Estacion'] = df_no2['FECHA'].apply(obtener_estacion)
df_pm25['Estacion'] = df_pm25['FECHA'].apply(obtener_estacion)

# 4. Seleccionar una estación de monitoreo, por ejemplo 'GAM'
estacion_monitoreo = 'GAM'

# 5. Crear el DataFrame combinado para esa estación
df_combined = pd.DataFrame({
    'PM10': df_pm10[estacion_monitoreo],
    'NO2': df_no2[estacion_monitoreo],
    'PM25': df_pm25[estacion_monitoreo]  #  variable objetivo (y)
})

# 6. Definir las variables predictoras (X) y la variable objetivo (y)
X = df_combined[['PM10', 'NO2']]   # Predictoras
y = df_combined['PM25']            # Objetivo

# 7. Dividir el conjunto de datos en entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 8. Mostrar el resultado de la división
print("Tamano del conjunto de entrenamiento:", X_train.shape)
print("Tamano del conjunto de prueba:", X_test.shape)

# 1. Crear el objeto StandardScaler
scaler = StandardScaler()

# 2. Ajustar el scaler a los datos de entrenamiento
X_train_scaled = scaler.fit_transform(X_train)

# 3. Transformar los datos de prueba
X_test_scaled = scaler.transform(X_test)

# Mostrar algunos resultados para asegurarnos de que se hizo correctamente
print("Datos de entrenamiento normalizados:\n", X_train_scaled[:5])
print("Datos de prueba normalizados:\n", X_test_scaled[:5])

# Paso a: Identificar valores atípicos utilizando IQR
Q1 = X_train.quantile(0.25)  # Primer cuartil
Q3 = X_train.quantile(0.75)  # Tercer cuartil
IQR = Q3 - Q1  # Rango intercuartílico

# Definir los límites para los valores atípicos
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR

# Identificar valores atípicos en el conjunto de entrenamiento
outliers = ((X_train < lower_bound) | (X_train > upper_bound))

# Mostrar cuántos valores atípicos existen
print(f"Valores atípicos en los datos de entrenamiento:\n{outliers.sum()}")

# Paso b: Eliminar valores atípicos
X_train_clean = X_train[~outliers.any(axis=1)]
y_train_clean = y_train[~outliers.any(axis=1)]

# Ver los datos después de la eliminación de valores atípicos
print(f"Tamaño del conjunto de datos después de eliminar los atípicos: {X_train_clean.shape}")

# Paso c: Documentar el impacto
print("\nImpacto:")
print(f"Total de valores atípicos eliminados: {outliers.any(axis=1).sum()}")

# a) Entrenamos un modelo RandomForestRegressor
rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)

# b) Extraemos las importancias de las características
importances = rf_model.feature_importances_

# c) Selección de las características más importantes
selected_features = X_train.columns[importances > 0.05]  # Umbral ajustado para ver más variables
print("Características seleccionadas:", selected_features)

# d) Descripción de los datos de NO2 y PM10
print("\nDescripción de los datos de NO2:")
print(df_combined['NO2'].describe())  # Descripción de los datos de NO2

print("\nDescripción de los datos de PM10:")
print(df_combined['PM10'].describe())  # Descripción de los datos de PM10

# 1. Cargar el archivo PM25 (por si no está cargado)
ruta_pm25 = r'C:\Users\Annna\OneDrive\Documentos\Inteligencia artificial\PrediccionCalidadDelAire\24RAMA\2024PM25.xls'
df_pm25 = pd.read_excel(ruta_pm25)

# 2. Convertir FECHA a datetime y ponerla como índice
df_pm25['FECHA'] = pd.to_datetime(df_pm25['FECHA'])
df_pm25.set_index('FECHA', inplace=True)


# 4. Para cada columna/estación hacemos la descomposición
for estacion in df_pm25.columns:
    # Evitar procesar la columna HORA
    if estacion == 'HORA':
        print(f"\nSaltando la columna: {estacion}")
        continue

    print(f"\nProcesando estación: {estacion}")

    # Revisar si la columna tiene valores nulos y manejarlos
    serie = df_pm25[estacion].dropna()  # Aquí eliminamos nulos. Si quieres interpolar avísame.

    # Si la serie tiene datos suficientes hacemos la descomposición
    if len(serie) > 365:
        result = seasonal_decompose(serie, model='additive', period=365)

        # Graficamos los resultados
        plt.figure(figsize=(12, 8))
        result.plot()
        plt.suptitle(f'Descomposición de PM2.5 en  {estacion}', fontsize=16)
        plt.tight_layout()
        plt.show()
    else:
        print(f" {estacion} no tiene suficientes datos para descomposición.")
        
# 1. Cargar el archivo PM25 (si no está cargado aún)
ruta_pm25 = r'C:\Users\Annna\OneDrive\Documentos\Inteligencia artificial\PrediccionCalidadDelAire\24RAMA\2024PM25.xls'
df_pm25 = pd.read_excel(ruta_pm25)

# 2. Convertir FECHA a datetime y ponerla como índice
df_pm25['FECHA'] = pd.to_datetime(df_pm25['FECHA'])
df_pm25.set_index('FECHA', inplace=True)


# 4. Filtramos columnas si hay alguna que no quieres (como HORA)
columnas_a_analizar = [col for col in df_pm25.columns if col.upper() != 'HORA']

# 5. Graficar ACF y PACF para cada estación, revisando los datos antes
for estacion in columnas_a_analizar:
    print(f"\nProcesando: {estacion}")

    # Tomar la serie sin valores nulos
    serie = df_pm25[estacion].dropna()

    # Validaciones 
    # 1. Que tenga más de 50 datos
    # 2. Que tenga más de 1 valor único (que no sea constante)
    if len(serie) > 50 and serie.nunique() > 1:
        plt.figure(figsize=(14, 6))

        # ACF
        plt.subplot(1, 2, 1)
        plot_acf(serie, ax=plt.gca(), lags=50)
        plt.title(f'ACF -  {estacion}')

        # PACF
        plt.subplot(1, 2, 2)
        plot_pacf(serie, ax=plt.gca(), lags=50, method='ywmle')  # método robusto

        plt.suptitle(f'ACF y PACF para {estacion}', fontsize=16)
        plt.tight_layout()
        plt.show()
    else:
        print(f" {estacion} no tiene suficientes datos o muestra valores constantes. Se omite del análisis.")
        
        
# Filtramos columnas si hay alguna que no quieres (como HORA)
columnas_a_analizar = [col for col in df_pm25.columns if col.upper() != 'HORA']

# Ajustar ARIMA para cada estación
for estacion in columnas_a_analizar:
    print(f"\nProcesando : {estacion}")

    # Tomar la serie sin nulos
    serie = df_pm25[estacion].dropna()

    if len(serie) > 50:  # Le ponemos un mínimo razonable para los cálculos
        # Creamos el modelo ARIMA(p,d,q) - en este caso, 1,1,1 como ejemplo
        modelo = ARIMA(serie, order=(1, 1, 1))

        # Entrenamos el modelo
        resultado = modelo.fit()

        # Mostramos el resumen con AIC, BIC, etc.
        print(f"\nResumen del modelo ARIMA para  {estacion}:")
        print(resultado.summary())

        # Diagnóstico de residuos, con validación de los residuos para evitar el error
        try:
            # Verificamos si la varianza de los residuos es suficientemente alta
            residuos = resultado.resid
            if residuos.var() > 1e-5:  # Solo graficamos si hay suficiente variabilidad
                resultado.plot_diagnostics(figsize=(15, 12))
                plt.suptitle(f'ARIMA -  {estacion}', fontsize=16)
                plt.tight_layout()
                plt.show()
            else:
                print(f" {estacion} no tiene suficiente variabilidad en los residuos para graficar.")
        
        except np.linalg.LinAlgError as e:
            print(f"Error en diagnóstico de residuos para la estación {estacion}: {e}")
            continue
    else:
        print(f" {estacion} no tiene suficientes datos para ARIMA.")
        
