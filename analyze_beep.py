
import sys
import numpy as np
from scipy.io import wavfile
from scipy.fft import rfft, rfftfreq
import matplotlib.pyplot as plt

def analyze_frequency(file_path):
    """
    Анализирует .wav файл, находит основную частоту и строит спектр.
    """
    try:
        # --- Загрузка аудиофайла ---
        sample_rate, data = wavfile.read(file_path)
        print(f"Файл '{file_path}' загружен. Частота: {sample_rate} Hz.")

        # Если звук в стерео, берем только один канал
        if data.ndim > 1:
            data = data[:, 0]

    except FileNotFoundError:
        print(f"Ошибка: Файл не найден по пути '{file_path}'")
        return
    except Exception as e:
        print(f"Ошибка при чтении файла: {e}")
        return

    # --- Анализ частот (FFT) ---
    N = len(data)
    if N == 0:
        print("Ошибка: аудиофайл пуст.")
        return

    data = np.frombuffer(data, dtype=np.int16).astype(np.float32)

    # Применяем оконную функцию для сглаживания
    data *= np.hamming(N)

    # Выполняем преобразование Фурье
    yf = np.abs(rfft(data))
    # Создаем массив частот для оси X
    xf = rfftfreq(N, 1 / sample_rate)

    print("\nСейчас откроется окно с графиком спектра. Закройте его, чтобы завершить программу.")

    # --- Визуализация ---
    plt.figure(figsize=(12, 6))
    plt.plot(xf, yf)
    plt.title(f"Спектр частот файла '{file_path}'")
    plt.xlabel("Частота (Гц)")
    plt.ylabel("Амплитуда")
    plt.grid(True)
    # Ограничим отображение до разумного предела (например, до 4000 Гц, т.к. частота дискретизации 8000)
    plt.xlim(0, sample_rate / 2)
    plt.show()

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Использование: python analyze_beep.py <путь_к_вашему_файлу.wav>")
        print("Пример: python analyze_beep.py beeps/beep_01.wav")
        sys.exit(1)

    file_to_analyze = sys.argv[1]
    analyze_frequency(file_to_analyze)
