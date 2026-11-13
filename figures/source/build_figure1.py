"""Build Figure 1 from its vector layout and the implemented sequence regions."""
from pathlib import Path
import argparse
import os
from lxml import etree as E
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.graphics import renderPDF
from svglib.svglib import svg2rlg
from svglib.fonts import register_font
import pypdfium2 as pdfium

HERE=Path(__file__).resolve().parent
NS='http://www.w3.org/2000/svg'

def add(root,tag,**attrs):
    return E.SubElement(root,'{'+NS+'}'+tag,**{k.replace('_','-'):str(v) for k,v in attrs.items()})

def build(out,font_dir):
    out.mkdir(parents=True,exist_ok=True)
    for name,file in [('Arial','arial.ttf'),('Arial-Bold','arialbd.ttf'),('Arial-Italic','ariali.ttf')]:
        pdfmetrics.registerFont(TTFont(name,str(font_dir/file)))
    pdfmetrics.registerFontFamily('Arial',normal='Arial',bold='Arial-Bold',italic='Arial-Italic')
    for weight,style,name,file in [('normal','normal','Arial','arial.ttf'),('bold','normal','Arial-Bold','arialbd.ttf'),('normal','italic','Arial-Italic','ariali.ttf')]:
        register_font('Arial',font_path=str(font_dir/file),weight=weight,style=style,rlgFontName=name)
    root=E.parse(str(HERE/'figure1_layout.svg')).getroot()
    x0,step=54.071,16.86613
    def region(start,end,y,label,color):
        x=x0+(start-1)*step;w=(end-start+1)*step
        add(root,'rect',x=f'{x:.3f}',y=y,width=f'{w:.3f}',height=4.2,fill=color)
        t=add(root,'text',x=f'{x+w/2:.3f}',y=y+11,text_anchor='middle',font_family='Arial',font_size=5.7,fill='#3A3A3A')
        t.text=label
    region(1,10,48.1,'5′ region (1–10)','#DCE7F1')
    region(13,20,48.1,'Seed (13–20)','#F8DFD6')
    region(21,23,48.1,'PAM (21–23)','#F8ECC8')
    region(7,16,64.3,'Middle (7–16)','#D5ECDE')
    region(17,23,64.3,'PAM-proximal (17–23)','#E1D9EF')
    stem=out/'figure1_framework'
    E.ElementTree(root).write(str(stem.with_suffix('.svg')),encoding='UTF-8',xml_declaration=True)
    drawing=svg2rlg(str(stem.with_suffix('.svg')))
    renderPDF.drawToFile(drawing,str(stem.with_suffix('.pdf')))
    doc=pdfium.PdfDocument(str(stem.with_suffix('.pdf')))
    for ext,dpi in [('png',240),('tiff',600)]:
        im=doc[0].render(scale=dpi/72).to_pil().convert('RGB')
        kw={'compression':'tiff_lzw'} if ext=='tiff' else {}
        im.save(stem.with_suffix('.'+ext),dpi=(dpi,dpi),**kw)
    doc.close()

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--out',type=Path,default=HERE.parent)
    p.add_argument('--font-dir',type=Path,default=Path(os.environ.get('WINDIR','C:/Windows'))/'Fonts')
    a=p.parse_args();build(a.out,a.font_dir)
