"""G3a (`g3_segmentacion_v1`): segmentador U-Net por fold y máscaras fuera de fold (OOF) de las 820 imágenes elegibles.

Mapeo v2, partición provisional `phash_le8_v1` (exploratoria, post hoc); conformidad académica pendiente. Este paquete NO abre el
test oficial ni las 180 normales. Reutiliza (sin modificar) `src/r4_v2` para partición, rutas y escritura que no sobrescribe, y
`src/segmentation/{unet,roi}.py`. torch, cv2 y PIL se importan de forma diferida: lo que no los necesita se prueba sin ellos.
"""
