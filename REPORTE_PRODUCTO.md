# Reporte del producto · GAIA Lab
Pronóstico de montaña a 3 km para apoyar el resguardo nocturno y la planificación del pastoreo en la puna.

> **Plantilla.** Copia este archivo a la raíz del repositorio de tu equipo y responde cada sección. Sé breve y concreto: 1 a 2 párrafos por pregunta, con números donde puedas. El jurado lo lee durante la preselección.

**Integrantes:** Yeremi Alexander Camacho Solís, Álvaro Rodrigo Ayon Chinga, Joseph Alí Huamanyauri Hernandez, Gabriel Dimas Gordillo Rodríguez**Producto:** [link al aplicativo o "se ejecuta en local, ver README"]
**Video pitch:** [link]

---

## 1. ¿Qué modelo o arquitectura usaron?

Describe tu modelo de downscaling: arquitectura (por ejemplo, CNN, U-Net, ConvLSTM, transformer), qué entradas usa (horas del GFS, relieve, variables extra creadas a partir de la data oficial), la función de pérdida y cómo validaste (qué corte temporal).

| | |
|---|---|
| Arquitectura | |
| Parámetros / tamaño de los pesos | |
| Score en el leaderboard público | |
| Score en tu validación local | |
| Tiempo de inferencia (24 horas de pronóstico) y hardware | |

## 2. ¿Para quién es y qué problema le resuelve?

El producto está dirigido a criadores de alpacas y ovinos de la puna. Antes de anochecer, el criador decide si resguarda el ganado en el cobertizo. Por la mañana, elige dónde llevarlo a pastar. Si considera quemar pastos, necesita saber si el fuego podría salirse de control. La app estima ese riesgo con los datos de las 13:00, hora de Lima, y la tarjeta de cobertizo ya muestra la hora prevista de la temperatura mínima. Una mala decisión puede dejar al ganado expuesto al frío, hacerle perder tiempo al mover los animales o dañar los pastos si el fuego se propaga.

**Lo que pidió el familiar.** En la consulta inicial, un familiar de un integrante pidió saber dónde pastorear para poder bajar el ganado si se espera una helada en la parte alta. El equipo propone incluir en el aviso la zona, el día y la acción sugerida.

## 3. ¿Cuál es la propuesta de valor del producto?

Por la tarde, el criador puede consultar si conviene resguardar el ganado esa noche y a qué ladera llevarlo al día siguiente, con el motivo de cada recomendación y un pronóstico en celdas de 3 km por lado, frente a los 27 km del GFS. Para el horizonte de siete días, la app muestra la curva de temperatura y el mapa de mínimas.

## 4. ¿Por qué tu producto es diferente a algo usual?

La app responde tres preguntas del criador: si conviene resguardar el ganado, dónde llevarlo a pastar y qué riesgo hay de que una quema se salga de control. Para el pastoreo, compara las zonas alta y baja con las salidas del modelo, pues en noches frías el aire frío puede acumularse en las partes bajas. La app indica la fuente de los umbrales e identifica cuáles fueron adaptados. La app muestra la acción sugerida y el motivo para que el criador pueda valorar esa información junto con lo que conoce del terreno.

## 5. ¿Cómo usa el producto la salida de tu modelo?

Explica el flujo de punta a punta: de dónde sale el GFS, cómo pasa por tu modelo y qué hace el producto con los mapas de 3 km. Si agregaste otras herramientas alrededor (LLM, traducción, alertas, otras fuentes de datos), di qué aporta cada una.

## 6. Limitaciones y próximos pasos
La app aún no envía avisos por SMS o WhatsApp.
¿Dónde falla o es menos confiable tu modelo o tu producto (zonas, variables, horas)? ¿Que podrías agregarle al producto? ¿Qué harían con más tiempo?
