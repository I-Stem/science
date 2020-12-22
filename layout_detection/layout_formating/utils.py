html_mathjax_header = '<!DOCTYPE html>\n<html lang="en">\n<head>\n<script type="text/javascript" src="https://cdn.mathjax.org/mathjax/latest/MathJax.js?config=TeX-AMS-MML_HTMLorMML"> </script>\n<script type="text/x-mathjax-config">\n   MathJax.Hub.Config({\n      tex2jax: { inlineMath: [["$","$"],["\\\\(","\\\\)"]] },\n      "HTML-CSS": {\n        linebreaks: { automatic: true, width: "container" }          \n      }              \n   });\n</script>\n</head>\n<body>\n<p>\n'
html_mathjax_footer = '</p>\n</body>\n</html>\n'

def bullet_remover( text, unordered_flag ) :
    if unordered_flag :
            text = text[1:] # removing first char i.e. unordered bullet
    else : # removing first word
        try :
            text =  text [ text.index(" ") : ]
        except: # if " " notfound then that line has just the list
            text = ""
    return text