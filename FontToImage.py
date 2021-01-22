from PIL import Image, ImageFont, ImageDraw
import glob
from fontTools.ttLib import TTFont


def has_glyph(font, glyph):
    for table in font['cmap'].tables:
        if ord(glyph) in table.cmap.keys():
            return True
    return False


def main():
    text = 'A'
    for font_path in glob.glob('fonts/*.*'):
        try:
            background = Image.new('RGB', (36, 36), color='white')
            if font_path.lower().endswith('ttf'):
                try_font = TTFont(font_path)
                if not has_glyph(try_font, text):
                    continue

            if font_path.lower().endswith('.ttf') or font_path.lower().endswith('.otf'):
                print(font_path)
                font = ImageFont.truetype(font_path, 28)
                image_editable = ImageDraw.Draw(background)
                image_editable.text((0, 0), text, (0, 0, 0), font=font)
                result_dir = 'result'
                background.save(result_dir + '/' + font_path.lower().split('fonts/')[1].split('.ttf')[0].split('.otf')[0] + '.jpg')
        except:
            print("That's Bizarre!")


if __name__ == "__main__":
    print('Starting the app...')
    main()
    print('Finished successfully!')
