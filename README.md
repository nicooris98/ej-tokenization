# Tokenizacion
Este ejemplo es para entender el stemming y lemmatization.

# Stemming
Recorta los sufijos de las palabras para obtener una raíz común (aunque no siempre sea una palabra real).
Método: Usa reglas simples o algoritmos como Porter, Snowball.
Precisión: Baja. Puede generar palabras inexistentes.
Velocidad: Alta. Es más rápido que la lematización.
Uso común: Cuando la velocidad importa más que la precisión (por ejemplo, búsquedas rápidas).
"running", "runs", "runner" → "run" (o incluso "runn")

# Lemmatization
Reduce las palabras a su forma canónica (lema), teniendo en cuenta la gramática y el contexto.
Método: Requiere un análisis morfológico y un diccionario.
Precisión: Alta. Solo produce palabras válidas del idioma.
Velocidad: Más lenta que el stemming.
Uso común: Cuando se requiere un procesamiento lingüístico más preciso (por ejemplo, en análisis de sentimientos o traducción automática).
"running", "ran" → "run"
"better" → "good" (más preciso que un stemmer)
