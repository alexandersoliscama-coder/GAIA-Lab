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

El flujo parte descargando el pronóstico horario global GFS (~27 km, grilla `17×13`) desde Open-Meteo con respaldo en caché local (`app/gfs_forecast.py`) para formar ventanas deslizantes de 3 horas (`[t-2, t-1, t]`). Nuestro modelo neuronal (`Net`, `models/modelo.pt`) combina esas ventanas con la codificación cíclica de la hora y el relieve estático SRTM (`data/static.npz`) para generar mapas horarios a **3 km** (`70×35` celdas) de humedad relativa, temperatura y viento (`RH, T, U, V`) en hora de Lima (`UTC - 5`), filtrados a la franja ganadera altoandina (`3800–4800 m`, Pulgar Vidal). Sobre esos mapas, `app/motor.py` calcula tres decisiones operativas para cinco sectores de Áncash: **(1) Cobertizo esta noche (`18:00–07:00`)**, evaluando la temperatura mínima nocturna (`≤ 0 °C` rojo según Senamhi-FAO / Da Motta; `≤ máx(2 °C, MAE)` ámbar) y el enfriamiento por viento (Nixon-Smith / FAO-56); **(2) Pastoreo mañana (`07:00–17:00`)**, comparando celdas vecinas altas y bajas dentro de `3800–4800 m` mediante temperatura real y enfriamiento diurno para detectar inversiones térmicas y evitar laderas con helada de madrugada (`00:00–07:00`); y **(3) Riesgo de quema (`13:00`)**, combinando `HR < 40 %` y el índice de Fosberg (`FFWI ≥ 50` en rojo).

Alrededor del modelo integramos dos componentes complementarios que refinan las alertas sin aumentar el consumo de API: **percentiles locales de enfriamiento (`p90` y `p97` en `models/chill_percentiles.npz`)**, calculados en seco (`R = 0`) sobre **111 noches del entrenamiento (18 feb – 30 jun 2024)** para adaptar el umbral de viento al microclima de cada celda sin disparar falsas alarmas cuando llueve; y la **precipitación horaria auxiliar del GFS en Open-Meteo**, consultada únicamente en las coordenadas de los 5 sectores (5 puntos en lugar de 221) para mostrar aparte la lluvia acumulada de las últimas 24 h como agravante de hipotermia en crías y contar días secos consecutivos (`< 1 mm/día`, criterio ETCCDI) en los últimos 15 días, activando precaución de quema al alcanzar `≥ 10` días secos seguidos.

## 6. Limitaciones y próximos pasos

El sistema presenta tres limitaciones técnicas principales que deben considerarse al interpretar las alertas: primero, aunque la salida se entrega en celdas de **3 km**, la resolución efectiva del downscaling desde GFS (~27 km) es menor que el espaciado de grilla (del orden de `~20 km` o `7Δx`, Skamarock, 2004), por lo que el drenaje de aire frío en quebradas muy estrechas puede variar localmente; segundo, los percentiles `p90`/`p97` del índice de enfriamiento están calibrados con **111 noches entre febrero y junio de 2024** (transición húmeda-seca), sin cubrir aún un ciclo multianual completo con el pico de heladas de julio y agosto; y tercero, como el modelo neuronal solo predice `RH, T, U, V`, la lluvia utilizada para el agravante de 24 h y los días secos proviene directamente del pronóstico global GFS (~27 km), que suaviza tormentas convectivas locales.

Como próximos pasos, proponemos validar en campo los umbrales provisionales (margen ámbar de `2 °C`, `10` días secos y franjas altitudinales de pastoreo) mediante entrevistas estructuradas con criadores de Áncash, recalibrar los percentiles con series históricas multianuales e incorporar un canal ligero de envío por SMS o WhatsApp para comunidades con conectividad intermitente.
