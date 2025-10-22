#!/usr/bin/env python3
"""
Générateur de QR code pour le portfolio
"""

import qrcode
from qrcode.image.styledpil import StyledPilImage
from qrcode.image.styles.moduledrawers import RoundedModuleDrawer
from qrcode.image.styles.colormasks import RadialGradiantColorMask
import os

def generate_portfolio_qr():
    """Génère un QR code pour le portfolio"""
    
    # URL du portfolio (à personnaliser)
    portfolio_url = "https://loick-dernoncourt.github.io/portfolio"
    
    # Configuration du QR code
    qr = qrcode.QRCode(
        version=1,
        error_correction=qrcode.constants.ERROR_CORRECT_L,
        box_size=10,
        border=4,
    )
    
    qr.add_data(portfolio_url)
    qr.make(fit=True)
    
    # Création de l'image avec style
    img = qr.make_image(
        image_factory=StyledPilImage,
        module_drawer=RoundedModuleDrawer(),
        color_mask=RadialGradiantColorMask(
            back_color=(255, 255, 255),
            center_color=(70, 130, 180),  # Bleu acier
            edge_color=(25, 25, 112)       # Bleu marine
        )
    )
    
    # Sauvegarde
    img.save("portfolio_qr.png")
    print(f"✅ QR code généré : portfolio_qr.png")
    print(f"🔗 URL : {portfolio_url}")
    
    return img

def generate_business_card_qr():
    """Génère un QR code pour carte de visite"""
    
    # Informations de contact
    contact_info = """BEGIN:VCARD
VERSION:3.0
FN:Loïck Dernoncourt
ORG:Data Scientist
TITLE:Senior Data Scientist
EMAIL:loick.dernoncourt@example.com
URL:https://loick-dernoncourt.github.io/portfolio
TEL:+33612345678
END:VCARD"""
    
    # Configuration du QR code
    qr = qrcode.QRCode(
        version=1,
        error_correction=qrcode.constants.ERROR_CORRECT_M,
        box_size=8,
        border=4,
    )
    
    qr.add_data(contact_info)
    qr.make(fit=True)
    
    # Création de l'image
    img = qr.make_image(
        image_factory=StyledPilImage,
        module_drawer=RoundedModuleDrawer(),
        color_mask=RadialGradiantColorMask(
            back_color=(255, 255, 255),
            center_color=(0, 100, 0),      # Vert
            edge_color=(0, 50, 0)         # Vert foncé
        )
    )
    
    # Sauvegarde
    img.save("business_card_qr.png")
    print(f"✅ QR code carte de visite généré : business_card_qr.png")
    
    return img

def generate_project_qr(project_name, project_url):
    """Génère un QR code pour un projet spécifique"""
    
    # Configuration du QR code
    qr = qrcode.QRCode(
        version=1,
        error_correction=qrcode.constants.ERROR_CORRECT_L,
        box_size=10,
        border=4,
    )
    
    qr.add_data(project_url)
    qr.make(fit=True)
    
    # Création de l'image
    img = qr.make_image(
        image_factory=StyledPilImage,
        module_drawer=RoundedModuleDrawer(),
        color_mask=RadialGradiantColorMask(
            back_color=(255, 255, 255),
            center_color=(255, 140, 0),    # Orange
            edge_color=(255, 69, 0)       # Rouge-orange
        )
    )
    
    # Sauvegarde
    filename = f"qr_{project_name.lower().replace(' ', '_')}.png"
    img.save(filename)
    print(f"✅ QR code projet '{project_name}' généré : {filename}")
    
    return img

if __name__ == "__main__":
    print("🚀 Génération des QR codes pour le portfolio")
    print("=" * 50)
    
    # QR code principal
    generate_portfolio_qr()
    
    # QR code carte de visite
    generate_business_card_qr()
    
    # QR codes pour projets spécifiques
    projects = [
        ("YOLO Detection", "https://github.com/loick-dernoncourt/yolov8-object-detection"),
        ("BERT Classification", "https://github.com/loick-dernoncourt/text-classification-bert"),
        ("Churn Prediction", "https://github.com/loick-dernoncourt/churn-prediction")
    ]
    
    for project_name, project_url in projects:
        generate_project_qr(project_name, project_url)
    
    print("\n🎉 Tous les QR codes ont été générés avec succès !")
    print("\n📱 Utilisation :")
    print("- portfolio_qr.png : QR code principal du portfolio")
    print("- business_card_qr.png : QR code pour carte de visite")
    print("- qr_*.png : QR codes pour projets spécifiques")
