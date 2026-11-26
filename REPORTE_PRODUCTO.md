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

El producto está dirigido a criadores de alpacas y ovinos de la puna. Por la noche, el criador decide si resguarda el ganado en el cobertizo. Por la mañana, elige dónde llevarlo a pastar. Si considera quemar pastos, necesita saber si el fuego podría salirse de control. Para esa tercera decisión, la app estima el riesgo con los datos de las 13:00, hora de Lima. Una mala decisión puede dejar al ganado expuesto al frío, hacerle perder tiempo al criador al mover los animales o dañar los pastos si el fuego se propaga.

**Lo que pidió el familiar.** Quiere saber dónde pastorear y, si se espera una helada en la parte alta, bajar el ganado. Pidió un aviso con una semana de anticipación por SMS o WhatsApp. **Propuesta del equipo.** Nos gustaría que el aviso indique la zona, el día y la acción sugerida, y que la app muestre la hora prevista de la temperatura mínima. El envío de avisos por SMS o WhatsApp todavía está por implementar.

### Por validar con el pastor

¿Cómo llama a sus zonas de pastoreo? ¿Qué señales usa para resguardar o mover al ganado? ¿Le sería útil saber la hora de la mínima? ¿Cuándo y por qué canal le conviene recibir el aviso? Al mostrarle el boceto, preguntarle qué haría, cuándo y por qué.

## 3. ¿Cuál es la propuesta de valor del producto?

Ganado Protegido ayuda al criador a preparar el resguardo nocturno y organizar el pastoreo. Para el sector consultado, muestra qué acción sugiere la app, cuándo se espera el frío y qué condiciones explican la recomendación. La consulta de hasta siete días permite anticipar cambios en el pastoreo y decidir con tiempo si se necesita preparar el cobertizo.

## 4. ¿Por qué tu producto es diferente a algo usual?

El producto usa el pronóstico global GFS, de unos 27 km, como entrada a un modelo que estima las condiciones en celdas de 3 km. En la consulta de pastoreo, la recomendación considera la helada de madrugada y el enfriamiento durante el día en las franjas disponibles del sector. La app muestra la acción sugerida y el motivo para que el criador pueda valorar esa información junto con lo que conoce del terreno.

## 5. ¿Cómo usa el producto la salida de tu modelo?

Explica el flujo de punta a punta: de dónde sale el GFS, cómo pasa por tu modelo y qué hace el producto con los mapas de 3 km. Si agregaste otras herramientas alrededor (LLM, traducción, alertas, otras fuentes de datos), di qué aporta cada una.

## 6. Limitaciones y próximos pasos

¿Dónde falla o es menos confiable tu modelo o tu producto (zonas, variables, horas)? ¿Que podrías agregarle al producto? ¿Qué harían con más tiempo?
