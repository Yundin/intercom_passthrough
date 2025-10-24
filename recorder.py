
import os
import signal
import sys
import time
import subprocess
import tempfile
from urllib.parse import urljoin

import requests
from dotenv import load_dotenv

# --- Глобальная переменная для хранения процесса FFmpeg ---
ffmpeg_process = None

class UfanetCam:
    """
    Класс для аутентификации и получения RTSP-ссылки на поток с камер Ufanet.
    """
    def __init__(self, username, password):
        self.username = username
        self.password = password
        self.session = requests.Session()
        self.session.headers.update({"Content-Type": "application/json"})

    def _get_access_token(self):
        """Шаг 1: Авторизация и получение JWT токена доступа."""
        print("Шаг 1: Получение токена доступа...")
        url = "https://dom.ufanet.ru/api/v1/auth/auth_by_contract/"
        data = {"contract": self.username, "password": self.password}
        try:
            response = self.session.post(url, json=data)
            response.raise_for_status()
            return response.json()["token"]["access"]
        except requests.RequestException as e:
            print(f"Ошибка на шаге 1: {e}")
            sys.exit(1)

    def _get_cam_server_url(self, access_token):
        """Шаг 2: Получение URL сервера камер."""
        print("Шаг 2: Получение URL сервера камер...")
        url = "https://dom.ufanet.ru/api/v0/contract/"
        self.session.headers.update({"Authorization": f"JWT {access_token}"})
        try:
            response = self.session.get(url)
            response.raise_for_status()
            return response.json()[0]["isp_org"]["cams_server"]["url"]
        except (requests.RequestException, IndexError, KeyError) as e:
            print(f"Ошибка на шаге 2: {e}")
            sys.exit(1)

    def _get_ucams_token(self, cam_server_url, access_token):
        """Шаг 3: Авторизация на сервере камер."""
        print("Шаг 3: Авторизация на сервере камер...")
        url = urljoin(cam_server_url, "/api/v0/auth/?ttl=20800")
        self.session.headers.update({"Authorization": f"JWT {access_token}"})
        try:
            response = self.session.post(url)
            response.raise_for_status()
            return response.json()["token"]
        except requests.RequestException as e:
            print(f"Ошибка на шаге 3: {e}")
            sys.exit(1)

    def _get_camera_info(self, cam_server_url, ucams_token):
        """Шаг 4: Получение информации о камере и токена для потока."""
        print("Шаг 4: Получение информации о камере...")
        url = urljoin(cam_server_url, "/api/v0/cameras/my/")
        self.session.headers.update({"Authorization": f"Bearer {ucams_token}"})
        data = {
            "order_by": "addr_asc",
            "fields": ["number", "server", "token_l"],
            "token_l_ttl": 86400,
            "page": 1,
            "page_size": 1,
        }
        try:
            response = self.session.post(url, json=data)
            response.raise_for_status()
            return response.json()["results"][0]
        except (requests.RequestException, IndexError, KeyError) as e:
            print(f"Ошибка на шаге 4: {e}")
            sys.exit(1)

    def get_rtsp_url(self):
        """
        Выполняет все шаги аутентификации и возвращает готовую RTSP-ссылку.
        """
        access_token = self._get_access_token()
        cam_server_url = self._get_cam_server_url(access_token)
        ucams_token = self._get_ucams_token(cam_server_url, access_token)
        cam_info = self._get_camera_info(cam_server_url, ucams_token)

        cam_id = cam_info["number"]
        token_l = cam_info["token_l"]
        domain = cam_info["server"]["domain"]

        rtsp_link = f"rtsp://{domain}/{cam_id}?token={token_l}&tracks=v1a1"
        return rtsp_link

def record_stream_ffmpeg(rtsp_url: str, output_filename: str):
    """
    Запускает FFmpeg для записи потока с аудио и видео.
    """
    global ffmpeg_process
    print(f"\nПодготовка к записи с помощью FFmpeg в файл: {output_filename}")

    command = [
        'ffmpeg',
        '-y',
        '-loglevel', 'error',
        '-rtsp_transport', 'tcp',
        '-i', rtsp_url,
        '-c:v', 'copy',
        '-c:a', 'aac',
        '-movflags', '+faststart',
        output_filename
    ]

    print("Команда для запуска:", " ".join(command))
    
    # Используем временный файл для логов stderr FFmpeg
    with tempfile.TemporaryFile(mode='w+', encoding='utf-8') as stderr_log:
        try:
            # Перенаправляем stderr в наш временный файл
            ffmpeg_process = subprocess.Popen(command, stdin=subprocess.PIPE, stderr=stderr_log)
            print(f"Начало записи. PID процесса FFmpeg: {ffmpeg_process.pid}")
            print("Нажмите Ctrl+C для остановки...")
            
            # wait() - ключевой момент. Он не трогает stdin/stderr, избегая ошибок.
            ffmpeg_process.wait()

        except FileNotFoundError:
            print("\nОшибка: FFmpeg не найден. Убедитесь, что он установлен и доступен в PATH.")
            sys.exit(1)
        except Exception as e:
            print(f"Произошла непредвиденная ошибка: {e}")

        # --- Анализ результата после завершения процесса ---
        if ffmpeg_process.returncode == 0:
            print(f"\nЗапись успешно завершена. Файл сохранен: {output_filename}")
        else:
            # Возвращаемся в начало временного файла, чтобы прочитать его
            stderr_log.seek(0)
            error_message = stderr_log.read().strip()
            if error_message:
                print(f"\nFFmpeg завершился с кодом {ffmpeg_process.returncode}. Ошибка:")
                print("--- STDERR ---")
                print(error_message)
            else:
                # Если код не 0, но в stderr пусто, это наш штатный выход по Ctrl+C
                print(f"\nЗапись остановлена пользователем. Файл сохранен: {output_filename}")

def signal_handler(sig, frame):
    """Обработчик сигналов для грациозной остановки FFmpeg путем отправки 'q'."""
    global ffmpeg_process
    if ffmpeg_process and ffmpeg_process.poll() is None:
        print("\nПолучен сигнал на завершение. Отправка команды 'q' процессу FFmpeg...")
        try:
            ffmpeg_process.stdin.write(b'q')
            ffmpeg_process.stdin.flush()
            ffmpeg_process.stdin.close()
        except (ValueError, IOError):
            # Игнорируем ошибки, если stdin уже закрыт
            pass
    else:
        sys.exit(0)

if __name__ == "__main__":
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    load_dotenv()
    username = os.getenv("UFA_USER")
    password = os.getenv("UFA_PASS")

    if not username or not password:
        print("Ошибка: Убедитесь, что UFA_USER и UFA_PASS заданы в .env файле.")
        sys.exit(1)

    cam = UfanetCam(username, password)
    rtsp_url = cam.get_rtsp_url()

    output_file = time.strftime('recording_%Y%m%d_%H%M%S.mp4')

    record_stream_ffmpeg(rtsp_url, output_file)
