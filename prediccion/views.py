from django.shortcuts import render
from django.http import JsonResponse
import pandas as pd
import joblib
import os
from django.views.decorators.csrf import csrf_exempt

# Ruta del modelo
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
modelo_path = os.path.join(BASE_DIR, "prediccion", "modelo.pkl")
model = joblib.load(modelo_path)

# Mapas de valores
sex_map = {"male": 1, "female": 0}
embarked_map = {"C": 0, "Q": 1, "S": 2}

# Vista principal
def index(request):
    return render(request, "index.html")

# Vista de predicción (POST)
@csrf_exempt  # permite POST sin problemas de CSRF si usamos JS
def predecir(request):
    if request.method == "POST":
        try:
            data = request.POST
            pclass = int(data["pclass"])
            sex = data["sex"].lower()
            age = float(data["age"])
            sibsp = int(data["sibsp"])
            parch = int(data["parch"])
            embarked = data["embarked"].upper()

            # Tomamos el precio que ingresó el usuario
            fare = float(data.get("fare", 0))

            # Creamos DataFrame con los datos de la persona
            nueva_persona = pd.DataFrame({
                "Pclass": [pclass],
                "Sex": [sex],
                "Age": [age],
                "SibSp": [sibsp],
                "Parch": [parch],
                "Fare": [fare],
                "Embarked": [embarked]
            })

            # Mapeamos valores categóricos
            nueva_persona["Sex"] = nueva_persona["Sex"].map(sex_map)
            nueva_persona["Embarked"] = nueva_persona["Embarked"].map(embarked_map)

            # Validamos que no haya valores nulos
            if nueva_persona.isnull().any().any():
                return JsonResponse({"error": "Valores de entrada no válidos"}, status=400)

            # Predicción usando el modelo
            pred = model.predict(nueva_persona)[0]
            resultado = "SOBREVIVE" if pred == 1 else "NO SOBREVIVE"

            return JsonResponse({"resultado": resultado})
        except Exception as e:
            return JsonResponse({"error": str(e)}, status=500)

    return JsonResponse({"error": "Método no permitido"}, status=405)

