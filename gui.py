"""
Módulo: gui.py
Interfaz gráfica usando PyQt6 para el algoritmo de Enfriamiento Simulado CVRP
"""

import sys
from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget,
    QVBoxLayout, QHBoxLayout, QFormLayout,
    QPushButton, QLabel, QSpinBox, QDoubleSpinBox, QTextEdit,
    QGroupBox, QProgressBar, QTableWidget, QTableWidgetItem,
    QTabWidget, QMessageBox, QFileDialog
)
from PyQt6.QtCore import Qt, QThread, pyqtSignal, QTimer
from PyQt6.QtGui import QFont, QColor, QPalette

import matplotlib
matplotlib.use("Qt5Agg")
from matplotlib.figure import Figure
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas

import numpy as np

from sa_cvrp_logic import SimulatedAnnealingCVRP, Punto, Ruta
from datos_dosquebradas import cargar_144_puntos_completos


class WorkerThread(QThread):
    """Thread para ejecutar el algoritmo sin bloquear la interfaz."""
    progreso_signal = pyqtSignal(float, float, float, int, int)  # progreso, mejor_costo, temperatura, iteracion, total
    resultado_signal = pyqtSignal(object, float, dict)  # rutas, costo, estadisticas
    error_signal = pyqtSignal(str)

    def __init__(self, algoritmo: SimulatedAnnealingCVRP):
        super().__init__()
        self.algoritmo = algoritmo
        self._is_running = True
        # Para no spamear la interfaz en cada iteración
        self._last_progress = -5.0

    def run(self):
        try:
            def callback_progreso(progreso, costo, temp, iteracion, total):
                # Throttling: solo emitimos si avanza al menos 1% o es el final
                if self._is_running:
                    if progreso >= 100 or progreso - self._last_progress >= 1.0:
                        self._last_progress = progreso
                        self.progreso_signal.emit(float(progreso), float(costo), float(temp), int(iteracion), int(total))

            rutas, costo, stats = self.algoritmo.ejecutar(callback_progreso)
            if self._is_running:
                self.resultado_signal.emit(rutas, costo, stats)
        except Exception as e:
            self.error_signal.emit(str(e))

    def stop(self):
        """Detiene el thread de manera segura."""
        self._is_running = False


class MapaCanvas(FigureCanvas):
    """Canvas para visualizar el mapa de rutas."""

    def __init__(self, parent=None):
        self.fig = Figure(figsize=(8, 6), dpi=100)
        self.ax = self.fig.add_subplot(111)
        super().__init__(self.fig)
        self.setParent(parent)

        self.fig.patch.set_facecolor("#2b3e50")
        self.ax.set_facecolor("#2f4255")

    def dibujar_rutas(self, rutas, puntos, punto_inicio_id):
        """Dibuja las rutas en el mapa con los ID de los puntos."""
        self.ax.clear()
        self.ax.set_facecolor("#2f4255")

        colores = ["#e74c3c", "#3498db", "#2ecc71", "#9b59b6",
                   "#f1c40f", "#e67e22", "#1abc9c", "#ecf0f1"]

        # Puntos de fondo en gris
        for punto in puntos:
            if punto.id != punto_inicio_id:
                self.ax.plot(
                    punto.longitud, punto.latitud, "o",
                    color="#7f8c8d", markersize=3, alpha=0.3, zorder=1
                )

        # Depósito
        punto_inicio = next(p for p in puntos if p.id == punto_inicio_id)
        self.ax.plot(
            punto_inicio.longitud, punto_inicio.latitud, "D",
            color="#f1c40f", markersize=12, markeredgecolor="black",
            zorder=10, label="Depósito"
        )

        # Rutas
        for idx, ruta in enumerate(rutas):
            color = colores[idx % len(colores)]
            xs, ys = [], []
            for pid in ruta.puntos:
                p = next(pp for pp in puntos if pp.id == pid)
                xs.append(p.longitud)
                ys.append(p.latitud)

            # Líneas de la ruta
            self.ax.plot(xs, ys, "-", color=color, linewidth=2, alpha=0.9, zorder=2)

            # Puntos interiores
            self.ax.plot(xs[1:-1], ys[1:-1], "o", color=color,
                         markersize=6, markeredgecolor="black", zorder=3)

            # Numerar con el **ID real** del punto (coincide con Detalle de Rutas)
            for lon, lat, pid in zip(xs[1:-1], ys[1:-1], ruta.puntos[1:-1]):
                self.ax.text(
                    lon, lat, str(pid),
                    ha="center", va="center",
                    fontsize=7, color="white",
                    zorder=4,
                    bbox=dict(boxstyle="round,pad=0.2",
                              facecolor=color, edgecolor="black", alpha=0.85)
                )

        self.ax.set_xlabel("Longitud", color="white")
        self.ax.set_ylabel("Latitud", color="white")
        self.ax.tick_params(colors="white")
        self.ax.set_title("Mapa de Rutas", color="white", fontsize=11)
        self.ax.grid(True, alpha=0.2, linestyle="--", color="#95a5a6")
        self.fig.tight_layout()
        self.draw()

    def limpiar(self):
        self.ax.clear()
        self.ax.set_facecolor("#2f4255")
        self.ax.text(
            0.5, 0.5,
            "Ejecute el algoritmo\npara ver las rutas",
            ha="center", va="center", color="white",
            fontsize=11, transform=self.ax.transAxes
        )
        self.ax.set_xticks([])
        self.ax.set_yticks([])
        self.draw()


class GraficoConvergencia(FigureCanvas):
    """Canvas para visualizar la convergencia del algoritmo."""

    def __init__(self, parent=None):
        self.fig = Figure(figsize=(8, 4), dpi=100)
        self.ax1 = self.fig.add_subplot(121)
        self.ax2 = self.fig.add_subplot(122)
        super().__init__(self.fig)
        self.setParent(parent)

        self.fig.patch.set_facecolor("#2b3e50")
        self.ax1.set_facecolor("#2f4255")
        self.ax2.set_facecolor("#2f4255")

    def dibujar_convergencia(self, historial_costos, historial_temperaturas):
        self.ax1.clear()
        self.ax2.clear()
        self.ax1.set_facecolor("#2f4255")
        self.ax2.set_facecolor("#2f4255")

        if not historial_costos or not historial_temperaturas:
            self.limpiar()
            return

        iters = list(range(len(historial_costos)))

        self.ax1.plot(iters, historial_costos, "-", linewidth=2)
        self.ax1.set_title("Mejor costo", color="white")
        self.ax1.set_xlabel("Iteración", color="white")
        self.ax1.set_ylabel("Costo (km)", color="white")
        self.ax1.grid(True, alpha=0.3, linestyle="--", color="#95a5a6")
        self.ax1.tick_params(colors="white")

        self.ax2.plot(iters, historial_temperaturas, "-", linewidth=2)
        self.ax2.set_title("Temperatura", color="white")
        self.ax2.set_xlabel("Iteración", color="white")
        self.ax2.set_ylabel("T", color="white")
        self.ax2.grid(True, alpha=0.3, linestyle="--", color="#95a5a6")
        self.ax2.tick_params(colors="white")

        self.fig.tight_layout()
        self.draw()

    def limpiar(self):
        self.ax1.clear()
        self.ax2.clear()
        for ax in (self.ax1, self.ax2):
            ax.set_facecolor("#2f4255")
            ax.text(
                0.5, 0.5,
                "Sin datos todavía",
                ha="center", va="center", color="white",
                fontsize=11, transform=ax.transAxes
            )
            ax.set_xticks([])
            ax.set_yticks([])
        self.draw()


class InterfazPrincipal(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Simulated Annealing CVRP - Dosquebradas")
        self.resize(1400, 700)

        self.puntos, self.punto_inicio_id = cargar_144_puntos_completos()
        self.algoritmo = None
        self.worker = None
        self.rutas_optimas = None

        self.configurar_estilo()
        self.init_ui()
        self.mostrar_info_inicial()

    # ----------------------------- Estilo global (tema oscuro)
    def configurar_estilo(self):
        QApplication.setStyle("Fusion")
        palette = QPalette()
        palette.setColor(QPalette.ColorRole.Window, QColor(44, 62, 80))
        palette.setColor(QPalette.ColorRole.WindowText, Qt.GlobalColor.white)
        palette.setColor(QPalette.ColorRole.Base, QColor(52, 73, 94))
        palette.setColor(QPalette.ColorRole.AlternateBase, QColor(44, 62, 80))
        palette.setColor(QPalette.ColorRole.ToolTipBase, Qt.GlobalColor.white)
        palette.setColor(QPalette.ColorRole.ToolTipText, Qt.GlobalColor.white)
        palette.setColor(QPalette.ColorRole.Text, Qt.GlobalColor.white)
        palette.setColor(QPalette.ColorRole.Button, QColor(52, 73, 94))
        palette.setColor(QPalette.ColorRole.ButtonText, Qt.GlobalColor.white)
        palette.setColor(QPalette.ColorRole.Highlight, QColor(26, 188, 156))
        palette.setColor(QPalette.ColorRole.HighlightedText, Qt.GlobalColor.black)
        QApplication.setPalette(palette)

    # ----------------------------------------------------------- UI general
    def init_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        main_layout = QHBoxLayout(central)
        main_layout.setContentsMargins(6, 6, 6, 6)
        main_layout.setSpacing(6)

        # --- LATERAL IZQUIERDO ---
        sidebar = self.crear_sidebar()
        sidebar.setFixedWidth(290)
        main_layout.addWidget(sidebar)

        # --- ZONA DERECHA ---
        right = QWidget()
        right_layout = QVBoxLayout(right)
        right_layout.setContentsMargins(0, 0, 0, 0)
        right_layout.setSpacing(4)

        self.tabs = QTabWidget()
        self.tabs.setDocumentMode(True)
        self.tabs.setStyleSheet("""
            QTabWidget::pane {
                border: 1px solid #34495e;
                background: #2f4255;
            }
            QTabBar::tab {
                background: #34495e;
                color: white;
                padding: 6px 18px;
                margin-right: 1px;
            }
            QTabBar::tab:selected {
                background: #1abc9c;
                color: black;
            }
            QTabBar::tab:hover {
                background: #3d566e;
            }
        """)

        # Tabs
        self.tab_mapa = self.crear_tab_mapa()
        self.tab_detalle = self.crear_tab_detalle()
        self.tab_convergencia = self.crear_tab_convergencia()
        self.tab_log = self.crear_tab_log()

        self.tabs.addTab(self.tab_mapa, "🗺️ Mapa de rutas")
        self.tabs.addTab(self.tab_detalle, "📋 Detalle de rutas")
        self.tabs.addTab(self.tab_convergencia, "📈 Convergencia")
        self.tabs.addTab(self.tab_log, "🧾 Log")

        right_layout.addWidget(self.tabs, stretch=1)
        main_layout.addWidget(right, stretch=1)

        # Timer para tiempo transcurrido
        self.tiempo_segundos = 0
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.actualizar_tiempo)

    # ------------------------------------------- Sidebar izquierda
    def crear_sidebar(self) -> QWidget:
        side = QWidget()
        layout = QVBoxLayout(side)
        layout.setContentsMargins(4, 4, 4, 4)
        layout.setSpacing(6)

        # Grupo parámetros
        group_params = QGroupBox("Parámetros Simulated Annealing")
        form = QFormLayout(group_params)

        self.spin_iter_max = QSpinBox()
        self.spin_iter_max.setRange(10, 10000)
        self.spin_iter_max.setValue(300)  # valor por defecto más rápido

        self.spin_temp_ini = QDoubleSpinBox()
        self.spin_temp_ini.setRange(10, 100000)
        self.spin_temp_ini.setDecimals(2)
        self.spin_temp_ini.setValue(5000.0)

        self.spin_enfriamiento = QDoubleSpinBox()
        self.spin_enfriamiento.setRange(0.80, 0.999)
        self.spin_enfriamiento.setDecimals(3)
        self.spin_enfriamiento.setSingleStep(0.005)
        self.spin_enfriamiento.setValue(0.990)

        self.spin_capacidad = QSpinBox()
        self.spin_capacidad.setRange(1000, 50000)
        self.spin_capacidad.setSingleStep(1000)
        self.spin_capacidad.setValue(25000)

        self.spin_vehiculos = QSpinBox()
        self.spin_vehiculos.setRange(1, 20)
        self.spin_vehiculos.setValue(14)

        form.addRow("Iteraciones Máximas:", self.spin_iter_max)
        form.addRow("Temperatura Inicial:", self.spin_temp_ini)
        form.addRow("Tasa de Enfriamiento:", self.spin_enfriamiento)
        form.addRow("Capacidad Vehículo (kg):", self.spin_capacidad)
        form.addRow("Número de Vehículos:", self.spin_vehiculos)

        layout.addWidget(group_params)

        # Grupo controles
        group_ctrl = QGroupBox("Controles")
        vctrl = QVBoxLayout(group_ctrl)

        self.btn_iniciar = QPushButton("🔥 Iniciar Enfriamiento Simulado")
        self.btn_iniciar.clicked.connect(self.ejecutar_algoritmo)

        self.btn_reiniciar = QPushButton("🔁 Reiniciar Vista")
        self.btn_reiniciar.clicked.connect(self.reiniciar_vista)

        self.btn_exportar = QPushButton("💾 Exportar Solución")
        self.btn_exportar.setEnabled(False)
        self.btn_exportar.clicked.connect(self.exportar_resultados)

        for btn in (self.btn_iniciar, self.btn_reiniciar, self.btn_exportar):
            btn.setMinimumHeight(30)
            vctrl.addWidget(btn)

        layout.addWidget(group_ctrl)

        # Grupo estadísticas
        group_stats = QGroupBox("Estadísticas")
        grid = QFormLayout(group_stats)

        self.lbl_iter = QLabel("0")
        self.lbl_mejor_costo = QLabel("N/A")
        self.lbl_costo_actual = QLabel("N/A")
        self.lbl_temp = QLabel("N/A")
        self.lbl_mejora = QLabel("0%")
        self.lbl_veh_usados = QLabel("0")
        self.lbl_tiempo = QLabel("00:00")

        grid.addRow("Iteración:", self.lbl_iter)
        grid.addRow("Mejor costo:", self.lbl_mejor_costo)
        grid.addRow("Costo actual:", self.lbl_costo_actual)
        grid.addRow("Temperatura:", self.lbl_temp)
        grid.addRow("Mejora:", self.lbl_mejora)
        grid.addRow("Vehículos usados:", self.lbl_veh_usados)
        grid.addRow("Tiempo:", self.lbl_tiempo)

        layout.addWidget(group_stats)

        # Barra de progreso (AHORA EN LA IZQUIERDA)
        self.barra_progreso = QProgressBar()
        self.barra_progreso.setValue(0)
        self.barra_progreso.setTextVisible(True)
        self.barra_progreso.setFixedHeight(22)
        self.barra_progreso.setStyleSheet("""
            QProgressBar {
                border: 1px solid #34495e;
                border-radius: 3px;
                background: #2f4255;
                text-align: center;
            }
            QProgressBar::chunk {
                background-color: #1abc9c;
            }
        """)
        layout.addWidget(self.barra_progreso)

        layout.addStretch(1)
        return side

    # ------------------------------------------- Pestañas derecha
    def crear_tab_mapa(self) -> QWidget:
        w = QWidget()
        v = QVBoxLayout(w)
        v.setContentsMargins(4, 4, 4, 4)
        v.setSpacing(4)
        self.mapa_canvas = MapaCanvas()
        self.mapa_canvas.limpiar()
        v.addWidget(self.mapa_canvas)
        return w

    def crear_tab_detalle(self) -> QWidget:
        w = QWidget()
        v = QVBoxLayout(w)
        v.setContentsMargins(4, 4, 4, 4)
        v.setSpacing(4)

        self.tabla_rutas = QTableWidget()
        self.tabla_rutas.setColumnCount(7)
        self.tabla_rutas.setHorizontalHeaderLabels([
            "Color", "Ruta", "Distancia (km)", "Carga (kg)",
            "Tiempo (h)", "Num. puntos", "Secuencia de IDs"
        ])
        self.tabla_rutas.horizontalHeader().setStretchLastSection(True)

        v.addWidget(self.tabla_rutas)
        return w

    def crear_tab_convergencia(self) -> QWidget:
        w = QWidget()
        v = QVBoxLayout(w)
        v.setContentsMargins(4, 4, 4, 4)
        v.setSpacing(4)
        self.grafico_convergencia = GraficoConvergencia()
        self.grafico_convergencia.limpiar()
        v.addWidget(self.grafico_convergencia)
        return w

    def crear_tab_log(self) -> QWidget:
        w = QWidget()
        v = QVBoxLayout(w)
        v.setContentsMargins(4, 4, 4, 4)
        v.setSpacing(4)

        self.texto_resumen = QTextEdit()
        self.texto_resumen.setReadOnly(True)
        self.texto_resumen.setStyleSheet(
            "QTextEdit { background:#1e272e; color:#ecf0f1; "
            "font-family:'Consolas'; font-size:9pt; }"
        )
        v.addWidget(self.texto_resumen)
        return w

    # ---------------------------------------------------------- Lógica
    def mostrar_info_inicial(self):
        info = f"""
SISTEMA CVRP - ENFRIAMIENTO SIMULADO

Puntos cargados: {len(self.puntos) - 1}
Punto de inicio: {self.puntos[0].nombre}
Demanda total: {sum(p.demanda for p in self.puntos[1:]):.0f} kg

Configure los parámetros en la barra izquierda y pulse
"Iniciar Enfriamiento Simulado".
"""
        self.texto_resumen.setPlainText(info)

    def ejecutar_algoritmo(self):
        if self.worker is not None and self.worker.isRunning():
            return

        # Preparar UI
        self.btn_iniciar.setEnabled(False)
        self.btn_exportar.setEnabled(False)
        self.barra_progreso.setValue(0)
        self.lbl_iter.setText("0")
        self.lbl_mejor_costo.setText("N/A")
        self.lbl_costo_actual.setText("N/A")
        self.lbl_temp.setText("N/A")
        self.lbl_mejora.setText("0%")
        self.lbl_veh_usados.setText("0")
        self.tiempo_segundos = 0
        self.lbl_tiempo.setText("00:00")
        self.timer.start(1000)
        self.texto_resumen.append("\n>>> Iniciando optimización...")

        # Instancia del algoritmo
        self.algoritmo = SimulatedAnnealingCVRP(
            puntos=self.puntos,
            capacidad_vehiculo=self.spin_capacidad.value(),
            num_vehiculos=self.spin_vehiculos.value(),
            punto_inicio_id=self.punto_inicio_id
        )
        self.algoritmo.temperatura_inicial = self.spin_temp_ini.value()
        self.algoritmo.factor_enfriamiento = self.spin_enfriamiento.value()
        self.algoritmo.iteraciones_por_temperatura = self.spin_iter_max.value()

        self.worker = WorkerThread(self.algoritmo)
        self.worker.progreso_signal.connect(self.actualizar_progreso)
        self.worker.resultado_signal.connect(self.mostrar_resultados)
        self.worker.error_signal.connect(self.mostrar_error)
        self.worker.finished.connect(self.on_worker_finished)
        self.worker.start()

    def on_worker_finished(self):
        self.btn_iniciar.setEnabled(True)
        self.timer.stop()

    def actualizar_tiempo(self):
        self.tiempo_segundos += 1
        m, s = divmod(self.tiempo_segundos, 60)
        self.lbl_tiempo.setText(f"{m:02d}:{s:02d}")

    def actualizar_progreso(self, progreso, mejor_costo, temperatura, iteracion, total_iteraciones):
        # progreso llega como 0–100
        self.barra_progreso.setValue(int(progreso))
        # Mostrar iteración actual de total (ej: "150 / 300")
        self.lbl_iter.setText(f"{iteracion} / {total_iteraciones}")
        self.lbl_mejor_costo.setText(f"{mejor_costo:.2f} km")
        self.lbl_costo_actual.setText(f"{mejor_costo:.2f} km")
        self.lbl_temp.setText(f"{temperatura:.2f}")

    def mostrar_resultados(self, rutas, costo_total, estadisticas):
        self.rutas_optimas = rutas
        self.btn_exportar.setEnabled(True)
        self.barra_progreso.setValue(100)

        mejora = ((estadisticas["costo_inicial"] - costo_total) /
                  estadisticas["costo_inicial"]) * 100

        self.lbl_mejora.setText(f"{mejora:.1f}%")
        self.lbl_veh_usados.setText(str(len(rutas)))

        iteraciones_totales = estadisticas.get('iteraciones_totales', 0)
        self.texto_resumen.append(
            f"\n>>> Optimización completa.\n"
            f"Iteraciones:   {iteraciones_totales}\n"
            f"Costo inicial: {estadisticas['costo_inicial']:.2f} km\n"
            f"Costo final:   {costo_total:.2f} km\n"
            f"Mejora:        {mejora:.2f}%\n"
        )

        self.mapa_canvas.dibujar_rutas(rutas, self.puntos, self.punto_inicio_id)
        self.grafico_convergencia.dibujar_convergencia(
            estadisticas["historial_costos"],
            estadisticas["historial_temperaturas"]
        )
        self.actualizar_tabla_rutas(rutas)

        QMessageBox.information(
            self,
            "Optimización completada",
            f"Costo final: {costo_total:.2f} km\nMejora: {mejora:.2f}%"
        )

    def actualizar_tabla_rutas(self, rutas):
        # Colores que coinciden con los del mapa
        colores = ["#e74c3c", "#3498db", "#2ecc71", "#9b59b6",
                   "#f1c40f", "#e67e22", "#1abc9c", "#ecf0f1"]
        
        self.tabla_rutas.setRowCount(len(rutas))
        for i, ruta in enumerate(rutas):
            num_puntos = len(ruta.puntos) - 2
            seq_str = " → ".join(str(p) for p in ruta.puntos)
            color_ruta = colores[i % len(colores)]

            # Columna 0: Color (cuadro de color)
            color_item = QTableWidgetItem("")
            color_item.setBackground(QColor(color_ruta))
            color_item.setFlags(color_item.flags() & ~Qt.ItemFlag.ItemIsEditable)
            self.tabla_rutas.setItem(i, 0, color_item)

            # Resto de columnas
            datos = [
                f"Ruta {i+1}",
                f"{ruta.distancia:.2f}",
                f"{ruta.carga:.0f}",
                f"{ruta.tiempo:.2f}",
                str(num_puntos),
                seq_str
            ]
            for col, valor in enumerate(datos, start=1):
                item = QTableWidgetItem(valor)
                if col != 6:  # La última columna (Secuencia) no se centra
                    item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
                self.tabla_rutas.setItem(i, col, item)

        # Ajustar ancho de la columna de color
        self.tabla_rutas.setColumnWidth(0, 50)
        self.tabla_rutas.resizeColumnsToContents()

    def mostrar_error(self, mensaje_error: str):
        self.texto_resumen.append(f"\n>>> ERROR: {mensaje_error}")
        QMessageBox.critical(self, "Error", mensaje_error)

    def exportar_resultados(self):
        if not self.rutas_optimas:
            QMessageBox.warning(self, "Sin datos", "No hay rutas para exportar.")
            return

        archivo, _ = QFileDialog.getSaveFileName(
            self, "Guardar resultados", "resultados_cvrp.txt",
            "Archivos de texto (*.txt);;Todos los archivos (*)"
        )
        if not archivo:
            return

        try:
            with open(archivo, "w", encoding="utf-8") as f:
                f.write(self.texto_resumen.toPlainText())
                f.write("\n\nDETALLE DE RUTAS\n")
                f.write("=" * 60 + "\n")
                for i, ruta in enumerate(self.rutas_optimas, 1):
                    f.write(f"RUTA {i}\n")
                    f.write(f"Distancia: {ruta.distancia:.2f} km\n")
                    f.write(f"Carga: {ruta.carga:.0f} kg\n")
                    f.write(f"Tiempo: {ruta.tiempo:.2f} h\n")
                    f.write("Secuencia: " + " -> ".join(str(p) for p in ruta.puntos) + "\n\n")
            QMessageBox.information(self, "OK", "Resultados exportados correctamente.")
        except Exception as e:
            QMessageBox.critical(self, "Error al exportar", str(e))

    def reiniciar_vista(self):
        self.mapa_canvas.limpiar()
        self.grafico_convergencia.limpiar()
        self.tabla_rutas.setRowCount(0)
        self.texto_resumen.clear()
        self.mostrar_info_inicial()
        self.barra_progreso.setValue(0)
        self.lbl_iter.setText("0")
        self.lbl_mejor_costo.setText("N/A")
        self.lbl_costo_actual.setText("N/A")
        self.lbl_temp.setText("N/A")
        self.lbl_mejora.setText("0%")
        self.lbl_veh_usados.setText("0")
        self.tiempo_segundos = 0
        self.lbl_tiempo.setText("00:00")


def main():
    app = QApplication(sys.argv)
    app.setFont(QFont("Segoe UI", 9))
    ventana = InterfazPrincipal()
    ventana.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
