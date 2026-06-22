import streamlit as st

from PIL import Image, ImageDraw

import numpy as np

import json

import pickle

import os

from tensorflow.keras.models import load_model

from tensorflow.keras.applications.mobilenet_v2 import preprocess_input



# --- CẤU HÌNH TRANG ---

st.set_page_config(page_title="Canteen AI Payment", page_icon="🍱", layout="wide")



st.markdown("""

    <style>

    .stApp { background-color: #FFF5F7; }

    h1 { color: #FF6B9D; text-align: center; }

    </style>

    """, unsafe_allow_html=True)



# Khởi tạo trạng thái để tránh mất dữ liệu khi nhấn nút hoặc re-run

if 'is_detected' not in st.session_state: st.session_state.is_detected = False

if 'bill_data' not in st.session_state: st.session_state.bill_data = []

# Đổi total_price thành base_total_price để cộng dồn tiền tùy chọn thêm

if 'base_total_price' not in st.session_state: st.session_state.base_total_price = 0

if 'total_calo' not in st.session_state: st.session_state.total_calo = 0

if 'last_file' not in st.session_state: st.session_state.last_file = None



# Trạng thái ghi nhận các món cần hiển thị tùy chọn

if 'has_trung_chien' not in st.session_state: st.session_state.has_trung_chien = False

if 'has_com_trang' not in st.session_state: st.session_state.has_com_trang = False

if 'has_thit_kho_trung' not in st.session_state: st.session_state.has_thit_kho_trung = False



# --- TẢI CẤU HÌNH TỌA ĐỘ ---

CONFIG_FILE = "coords.json"



def load_coords():

    if os.path.exists(CONFIG_FILE):

        try:

            with open(CONFIG_FILE, 'r') as f:

                return json.load(f)

        except Exception:

            pass

    # Trả về tọa độ mặc định ban đầu nếu chưa có file json

    return {

        "Ngăn 1": [411, 54, 383, 400],

        "Ngăn 2": [752, 119, 396, 318],

        "Ngăn 3": [1200, 132, 290, 322],

        "Ngăn 4": [1031, 608, 396, 319],

        "Ngăn 5": [457, 586, 342, 327]

    }



current_coords = load_coords()



# --- SIDEBAR: CẤU HÌNH TỌA ĐỘ VÀ TỰ ĐỘNG LƯU ---

st.sidebar.title("🛠 Tùy chỉnh tọa độ")

new_coords = {}



for name in ["Ngăn 1", "Ngăn 2", "Ngăn 3", "Ngăn 4", "Ngăn 5"]:

    with st.sidebar.expander(f"📍 {name}"):

        box_vals = current_coords.get(name, [0, 0, 100, 100])

        x = st.number_input(f"{name} X", value=int(box_vals[0]), key=f"x_{name}")

        y = st.number_input(f"{name} Y", value=int(box_vals[1]), key=f"y_{name}")

        w = st.number_input(f"{name} Rộng (W)", value=int(box_vals[2]), key=f"w_{name}")

        h = st.number_input(f"{name} Cao (H)", value=int(box_vals[3]), key=f"h_{name}")

        new_coords[name] = [x, y, w, h]



if st.sidebar.button("💾 Lưu tọa độ cố định"):

    with open(CONFIG_FILE, 'w') as f:

        json.dump(new_coords, f, indent=4)

    st.sidebar.success("Đã lưu tọa độ vào file coords.json!")

    st.rerun()



# --- TẢI MÔ HÌNH VÀ TÀI NGUYÊN AI ---

@st.cache_resource

def load_resources():

    model = load_model('model_mobilenet.h5')

    with open('labels_cnn.pkl', 'rb') as f:

        labels = {v: k for k, v in pickle.load(f).items()}

    with open('menu.json', 'r', encoding='utf-8') as f:

        menu = json.load(f)

    return model, labels, menu



try:

    model, labels, menu = load_resources()

except Exception as e:

    st.error(f"Lỗi tải file (kiểm tra lại file model/menu/labels): {e}")

    st.stop()



# --- GIAO DIỆN CHÍNH ---

st.title('Canteen của BA')

uploaded_file = st.file_uploader("Chọn ảnh khay cơm...", type=["jpg", "png", "jpeg"])



if uploaded_file != st.session_state.last_file:

    st.session_state.is_detected = False

    st.session_state.last_file = uploaded_file



if uploaded_file:

    img = Image.open(uploaded_file).convert('RGB')

    coords = new_coords

   

    img_display = img.copy()

    draw = ImageDraw.Draw(img_display)

    for name, (x, y, w, h) in coords.items():

        draw.rectangle([x, y, x+w, y+h], outline="yellow", width=5)

    st.image(img_display, caption="Ảnh khay cơm đã xác định vùng", use_container_width=True)



    # --- KÍCH HOẠT NHẬN DIỆN VÀ TÍNH TOÁN ---

    if st.button("✨ Tính tiền và Calo"):

        with open(CONFIG_FILE, 'w') as f:

            json.dump(coords, f, indent=4)

           

        bill_data = []

        base_total_price = 0

        total_calo = 0

       

        # Reset cờ nhận diện món ăn đặc biệt

        has_tc = False

        has_ct = False

        has_tkt = False

       

        for name, (x, y, w, h) in coords.items():

            crop = img.crop((x, y, x+w, y+h))

            crop_resized = crop.resize((128, 128))

           

            img_arr = np.array(crop_resized)

            img_arr = np.expand_dims(img_arr, axis=0)

            img_arr = preprocess_input(img_arr)

           

            predictions = model.predict(img_arr, verbose=0)

           

            sorted_indices = np.argsort(predictions[0])[::-1]

            top1_idx = sorted_indices[0]

            top2_idx = sorted_indices[1]

           

            top1_key = labels[top1_idx]

            top2_key = labels[top2_idx]

            conf1 = predictions[0][top1_idx] * 100

           

            dish_key = top1_key

           

            # 🌟 BỘ LỌC HEURISTIC

            if top1_key == 'trung_chien' and top2_key == 'suon_nuong' and conf1 < 70:

                dish_key = 'suon_nuong'

            elif top1_key == 'suon_nuong' and top2_key == 'trung_chien' and conf1 < 55:

                dish_key = 'trung_chien'

            elif top1_key == 'com_trang' and top2_key == 'trung_chien' and conf1 < 65:

                if w * h < 130000:  

                    dish_key = 'trung_chien'

            elif top1_key == 'com_trang' and top2_key == 'suon_nuong' and conf1 < 60:

                if w * h < 130000:

                    dish_key = 'suon_nuong'



            # Ghi nhận nếu các món đặc biệt xuất hiện

            if dish_key == 'trung_chien': has_tc = True

            if dish_key == 'com_trang': has_ct = True

            if dish_key == 'thit_kho_trung': has_tkt = True



            # Trích xuất dữ liệu từ menu

            dish_info = menu.get(dish_key, {"ten": dish_key, "gia": 0, "calo": 0})

           

            base_total_price += dish_info.get('gia', 0)

            total_calo += dish_info.get('calo', 0)

           

            bill_data.append({

                "Ngăn": name,

                "Món": dish_info.get('ten', dish_key),

                "Giá": f"{dish_info.get('gia', 0):,}đ",

                "Calo": f"{dish_info.get('calo', 0)} kcal"

            })



        # Lưu trữ trạng thái vào Session State

        st.session_state.bill_data = bill_data

        st.session_state.base_total_price = base_total_price

        st.session_state.total_calo = total_calo

        st.session_state.has_trung_chien = has_tc

        st.session_state.has_com_trang = has_ct

        st.session_state.has_thit_kho_trung = has_tkt

        st.session_state.is_detected = True

        st.rerun()



    # --- HIỂN THỊ KẾT QUẢ VÀ HÓA ĐƠN ---

    if st.session_state.is_detected:

        st.subheader("📋 Hóa đơn chi tiết")

        st.table(st.session_state.bill_data)

       

        # --- TÙY CHỌN THÊM (OPTIONS) ---

        extra_fee = 0

        if st.session_state.has_trung_chien or st.session_state.has_com_trang or st.session_state.has_thit_kho_trung:

            st.markdown("### 📝 Tùy chọn món thêm")

            col_opt1, col_opt2, col_opt3 = st.columns(3)

           

            if st.session_state.has_trung_chien:

                with col_opt1:

                    is_trung_thit = st.checkbox("🍳 Trứng chiên thịt (+8,000đ)")

                    if is_trung_thit: extra_fee += 8000

                   

            if st.session_state.has_com_trang:

                with col_opt2:

                    is_com_them = st.checkbox("🍚 Cơm thêm (+2,000đ)")

                    if is_com_them: extra_fee += 2000

                   

            if st.session_state.has_thit_kho_trung:

                with col_opt3:

                    extra_eggs = st.number_input("🥚 Số trứng thêm (+6,000đ/trứng)", min_value=0, step=1)

                    extra_fee += extra_eggs * 6000

       

        # Cập nhật tổng tiền (bao gồm món chính + tùy chọn thêm)

        total_before_discount = st.session_state.base_total_price + extra_fee

       

        col1, col2 = st.columns(2)

        col1.metric("Tổng tiền (chưa giảm)", f"{total_before_discount:,} VNĐ")

        col2.metric("Tổng Calo", f"{st.session_state.total_calo} kcal")

       

        st.divider()



        # --- TÍNH NĂNG VOUCHER ---

        st.subheader("🎟️ Mã giảm giá")

        voucher_code = st.text_input("Nhập mã giảm giá (nếu có):").strip().lower()

       

        final_price = total_before_discount

       

        if voucher_code == "bonvatrum":

            discount_amount = int(total_before_discount * 0.1)  # Giảm 10%

            final_price = total_before_discount - discount_amount

            st.success(f"🎉 Áp dụng thành công mã giảm 10%! Bạn được giảm **{discount_amount:,} VNĐ**.")

        elif voucher_code != "":

            st.error("❌ Mã giảm giá không hợp lệ.")

           

        st.metric("💰 Số tiền cần thanh toán", f"{final_price:,} VNĐ")

        st.divider()



        # --- PHƯƠNG THỨC THANH TOÁN ---

        st.subheader("💳 Phương thức thanh toán")

        payment_method = st.radio("Chọn cách thanh toán:", ["Tiền mặt", "Chuyển khoản QR"])

       

        if payment_method == "Chuyển khoản QR":

            BANK_ID = "VCB"

            ACCOUNT_NO = "1047413581"

            # Cập nhật số tiền cần quét bằng final_price (đã tính topping + áp mã giảm giá)

            qr_url = f"https://img.vietqr.io/image/{BANK_ID}-{ACCOUNT_NO}-compact2.png?amount={final_price}&addInfo=ThanhToanCanteen&accountName=CANTEEN"

            st.info("Quét mã QR để thanh toán (Đã tự động tính gộp tiền tùy chọn thêm):")

            st.image(qr_url, caption="Mã QR thanh toán", width=300)

           

        else:

            st.warning("Nhập số tiền khách đưa:")

            customer_cash = st.number_input("Số tiền khách đưa (VNĐ):", min_value=0, step=1000)

           

            if customer_cash > 0:

                if customer_cash >= final_price:

                    change = customer_cash - final_price

                    st.success(f"✅ Tiền thối lại cho khách: **{change:,} VNĐ**")

                    st.balloons()

                else:

                    st.error(f"❌ Số tiền khách đưa chưa đủ! Còn thiếu: **{final_price - customer_cash:,} VNĐ**") 