# ☀️ Inspección automática de paneles solares

App de Streamlit que clasifica el estado de un panel solar a partir de una foto y recomienda la acción de mantenimiento.

**TP2 – Análisis Predictivo Avanzado (ITBA)**

## Problema de negocio

Los parques solares tienen miles de paneles cuya eficiencia cae por suciedad (polvo, excremento de aves, nieve) o por daños (eléctricos o físicos). Revisar manualmente todas las fotos tomadas por drones no escala. El modelo clasifica automáticamente cada panel y prioriza el trabajo de los inspectores.

## Modelo

- **Arquitectura:** EfficientNetB0 preentrenado en ImageNet + *fine-tuning* (Keras / TensorFlow).
- **Clases:** Limpio, Polvo, Excremento de aves, Nieve, Daño eléctrico, Daño físico.
- **Resultados en test:** accuracy 0,746 · F1 macro 0,747 · Kappa 0,687.
- Ningún panel dañado fue clasificado como limpio en test.

## Lógica de decisión

1. El modelo estima la probabilidad de cada clase.
2. Las clases se agrupan en 3 acciones:

| Acción | Clases |
|---|---|
| **Nada** | Limpio |
| **Limpieza** | Polvo, Excremento de aves, Nieve |
| **Técnico** | Daño eléctrico, Daño físico |

3. Se elige la acción de **mínimo costo esperado** según una matriz de costos.
4. Si la confianza es menor al umbral, la imagen se **deriva a revisión humana**.

## Impacto de negocio

Con capacidad de revisión humana limitada (escenario realista en parques grandes), usar el modelo para priorizar qué fotos revisar reduce el costo de mantenimiento entre un **55% y un 66%** respecto de revisar una muestra al azar. Revisando solo el **30%** de las fotos (los casos más dudosos), el costo queda apenas un 11% por encima de revisar el 100% a mano.

## Archivos

- `app.py`: aplicación de Streamlit.
- `modelo_paneles.keras`: modelo entrenado.
- `config.json`: clases, acciones, matriz de costos y umbral.
- `requirements.txt`: dependencias.

## Ejecutar localmente

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Limitaciones

- Dataset chico (782 imágenes tras eliminar duplicados), recolectado de internet.
- El modelo es sensible a imágenes desenfocadas: en producción conviene un control de calidad de imagen previo.

## Dataset

*Solar Panel Images Clean and Faulty Images* (Kaggle).
