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

¿Quién es tu usuario concreto (agricultor, guía de montaña, municipalidad…)? ¿Qué decisión/accion/etc toma hoy sin buena información y qué decide/toma/etc distinto con tu producto?

## 3. ¿Cuál es la propuesta de valor del producto?

En una o dos frases: qué gana el usuario al usar tu producto. Si puedes, cuantifícalo (tiempo, dinero, riesgo evitado, alcance).

## 4. ¿Por qué tu producto es diferente a algo usual?

¿Qué lo distingue de una app de clima común o de un mapa con el pronóstico? ¿Qué hace que el usuario lo prefiera?

## 5. ¿Cómo usa el producto la salida de tu modelo?

El flujo descarga el pronóstico horario global GFS (~27 km, grilla `17×13`) desde Open-Meteo con caché local (`app/gfs_forecast.py`) y arma ventanas de 3 horas (`[t-2, t-1, t]`). Nuestro modelo neuronal (`Net`, `models/modelo.pt`) combina esas ventanas con la codificación cíclica de la hora y el relieve estático SRTM (`data/static.npz`) para generar mapas horarios a **3 km** (`70×35` celdas) de `RH, T, U, V` en hora de Lima (`UTC - 5`), filtrados por defecto a `3800–4800 m` (rango configurable en la app; Pulgar Vidal ubica la puna entre `4000` y `4800 m`). Sobre esos mapas, `app/motor.py` genera tres recomendaciones para cinco sectores de Áncash:

* **Cobertizo esta noche (`18:00–07:00`):** evalúa la temperatura mínima nocturna (`≤ 0 °C` rojo según Senamhi-FAO / Da Motta; `≤ máx(2 °C, MAE)` ámbar) y el índice de enfriamiento (Nixon-Smith / FAO-56).
* **Pastoreo mañana (`07:00–17:00`):** compara la zona alta con la zona baja dentro del rango altitudinal sin asumir que abajo hace menos frío, evaluando la temperatura real y el enfriamiento diurno tras descartar laderas con helada de madrugada (`00:00–07:00`).
* **Riesgo de quema (`13:00`):** combina `HR < 40 %` con el índice de Fosberg (`FFWI ≥ 50` en rojo).

Además, la aplicación muestra la comparación directa entre el pronóstico grueso GFS (~27 km) y el downscaling a 3 km del modelo, y permite ejecutar todo el flujo tanto en tiempo real como con fechas pasadas. Alrededor del modelo sumamos dos fuentes complementarias: **percentiles locales (`p90` y `p97` en `models/chill_percentiles.npz`)** del **índice de enfriamiento** (que combina temperatura y viento) calculados en seco (`R = 0`) sobre 111 noches de entrenamiento (18 feb – 30 jun 2024) para calibrar la alerta al microclima local sin falsas alarmas por lluvia; y la **precipitación horaria auxiliar del GFS** consultada solo en los 5 sectores para mostrar aparte la lluvia acumulada de 24 h como agravante y contar días secos consecutivos (`< 1 mm/día`, criterio ETCCDI) en los últimos 15 días (precaución de quema desde `≥ 10` días).

## 6. Limitaciones y próximos pasos

El sistema tiene cuatro limitaciones principales: primero, aunque entrega celdas de **3 km**, la resolución efectiva del downscaling desde GFS (~27 km) es menor que el espaciado de grilla (del orden de `~20 km` o `7Δx`, Skamarock, 2004), por lo que quebradas estrechas pueden variar localmente; segundo, los percentiles `p90`/`p97` del índice de enfriamiento están calibrados con **111 noches entre febrero y junio de 2024**, sin cubrir julio ni agosto; tercero, la lluvia usada para el agravante de 24 h y los días secos proviene directamente del GFS (~27 km) y no del modelo neuronal; y cuarto, a nivel de producto, las recomendaciones cubren solo **esta noche y mañana**, mientras que el criador solicitó avisos con una semana de anticipación.

Como próximos pasos, proponemos validar en campo los umbrales provisionales (margen ámbar de `2 °C`, `10` días secos y franjas altitudinales de pastoreo) junto a criadores de Áncash, extender la calibración de percentiles a un año completo e incorporar alertas tempranas con horizonte de 7 días.