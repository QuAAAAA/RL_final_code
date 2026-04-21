from gradio_client import Client, handle_file
import shutil

# 解決 SSL 憑證問題
client = Client("https://140.113.30.139:5003/", ssl_verify=False)


# you can change model here in accordance with the choices on my gradio webui
# 1️⃣ 切換台語模型
result = client.predict(
    # ✔ e.g., "pretrained_For_Selection/華語A3模型",
    # 台語:台語模型
    # 英語:A3模型
    # 華語:華語A3模型
    "pretrained_For_Selection/台語模型",
    api_name="/change_model"
)


# 2️⃣ 合成語音
try:
    audio = client.predict(
        # ✔ text you want to synthesize, e.g., "Hello, world!"
        tts_text = "lan2 tsit4 kai2 tio7 beh4 tui3 tam7-tsui2-ho5 khau2 jip8--khi3,lai5 tsau2-tshue7 pat4-li2 kap4 tam7-tsui2 tsi1-kan1 e5 siang1-siann5 koo3-su7", 
        mode_checkbox_group = "3s極速覆刻", 
        # ✔ text which is transcribed from prompt audio, e.g., "Hello, world!"
        prompt_text = "ai3 tsu3- i3 an1- tsuan5 --ooh4 , a1- kong1 tshue1 tian7- hong1 , lin2 u7 oh8 --khi2- lai5 ah8 bo5 ?", 
        # ✔ audio path that you want to clone.
        prompt_wav_upload = handle_file("./examples/libritts/sarcvoice2/文化相放伴_ep080_085_測試集.wav"),
        # prompt_text_record 
        prompt_wav_record = None, 
        # seed for randomness from 0 to 100000000
        seed = 51234, 
        # 1.0 is normal speed
        speed = 1.0, 
        # ✔ whether to enable translation
        enable_translation = False, 
        api_name="/generate"
    )

    filename = "taigi_example.wav" # you can change the filename to save the synthesized audio locally
    shutil.copy(audio, filename) 
    print(f"Audio saved as {filename}")

except Exception as e:
    print(f"合成失敗，請檢查伺服器狀態或權限: {e}")