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

El producto está dirigido a criadores de alpacas y ovinos de la puna. Antes de anochecer, el criador decide si lleva el ganado al cobertizo o lo deja en el corral. Por la mañana, decide si lleva sus animales a pastar a la parte alta o a una zona más baja. Para elegir necesita saber en qué lugar y a qué hora se espera el frío. Si se pronostica helada arriba, quiere saber si conviene bajar los animales.

En la consulta inicial, un familiar de un integrante pidió saber dónde pastorear y recibir el aviso con una semana de anticipación, por SMS o WhatsApp. La entrevista con el pastor permitirá confirmar los nombres locales de las zonas, el mejor momento para avisar y el canal que le resulta más práctico.

## 3. ¿Cuál es la propuesta de valor del producto?

Ganado Protegido ayuda al criador a preparar el resguardo nocturno y organizar el pastoreo. Para el sector consultado, muestra qué acción sugiere la app, cuándo se espera el frío y qué condiciones explican la recomendación. La consulta de hasta siete días permite anticipar cambios en el pastoreo y decidir con tiempo si se necesita preparar el cobertizo.

## 4. ¿Por qué tu producto es diferente a algo usual?

El producto usa el pronóstico global GFS, de unos 27 km, como entrada a un modelo que estima las condiciones en celdas de 3 km. En la consulta de pastoreo, la recomendación considera la helada de madrugada y el enfriamiento durante el día en las franjas disponibles del sector. La app muestra la acción sugerida y el motivo para que el criador pueda valorar esa información junto con lo que conoce del terreno.

## 5. ¿Cómo usa el producto la salida de tu modelo?

Explica el flujo de punta a punta: de dónde sale el GFS, cómo pasa por tu modelo y qué hace el producto con los mapas de 3 km. Si agregaste otras herramientas alrededor (LLM, traducción, alertas, otras fuentes de datos), di qué aporta cada una.

## 6. Limitaciones y próximos pasos

¿Dónde falla o es menos confiable tu modelo o tu producto (zonas, variables, horas)? ¿Que podrías agregarle al producto? ¿Qué harían con más tiempo?
