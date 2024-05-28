#orgによる特長点検出

import cv2
import os

# 画像の絶対パスを指定
image_path = '画像'

# 画像ファイルが存在するか確認
if not os.path.exists(image_path):
    print(f"Error: Image file does not exist at path {image_path}")
else:
    # 画像を読み込む
    image = cv2.imread(image_path, 0)

    # 画像が正しく読み込まれたか確認
    if image is None:
        print(f"Error: Image at path {image_path} could not be loaded.")
    else:
        # ORBオブジェクトの生成
        orb = cv2.ORB_create()

        # 特徴点と記述子の検出
        keypoints, descriptors = orb.detectAndCompute(image, None)

        # 画像に特徴点を描画
        image_with_keypoints = cv2.drawKeypoints(image, keypoints, None, color=(0, 255, 0))

        # 結果を表示
        cv2.imshow('ORB Keypoints', image_with_keypoints)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

