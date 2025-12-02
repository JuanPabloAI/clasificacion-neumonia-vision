# Despliegue en GitHub Pages

Este directorio contiene el blog post técnico del proyecto, configurado para ser publicado en GitHub Pages.

## 📝 Contenido

- `index.md`: Blog post completo con análisis técnico del proyecto
- `_config.yml`: Configuración de Jekyll para GitHub Pages

## 🚀 Instrucciones de Publicación

### 1. Configurar GitHub Pages en el Repositorio

1. Ve a tu repositorio en GitHub
2. Navega a **Settings** → **Pages**
3. En **Source**, selecciona:
   - Branch: `main` (o la rama principal)
   - Folder: `/docs`
4. Click en **Save**

### 2. Verificar Publicación

Después de 1-2 minutos, el sitio estará disponible en:

```
https://<usuario>.github.io/<nombre-repositorio>/
```

Por ejemplo:
```
https://JuanPabloAI.github.io/clasificacion-neumonia-vision/
```

### 3. Personalizar el Tema (Opcional)

El blog usa el tema **Cayman** por defecto. Para cambiar el tema:

1. Edita `_config.yml`
2. Cambia la línea `theme:` a uno de estos temas soportados:
   - `jekyll-theme-minimal`
   - `jekyll-theme-slate`
   - `jekyll-theme-architect`
   - `jekyll-theme-cayman`
   - `jekyll-theme-dinky`
   - `jekyll-theme-hacker`
   - `jekyll-theme-leap-day`
   - `jekyll-theme-merlot`
   - `jekyll-theme-midnight`
   - `jekyll-theme-modernist`
   - `jekyll-theme-tactile`
   - `jekyll-theme-time-machine`

### 4. Agregar Imágenes

Para incluir las imágenes de resultados:

1. Crea una carpeta `docs/assets/images/`
2. Copia las imágenes desde `results/figures/`:
   - `pca_2d_visualization.png`
   - `roc_curves.png`
   - Confusion matrices
   - Otros gráficos relevantes
3. Actualiza las rutas en `index.md`:

```markdown
![Visualización PCA](assets/images/pca_2d_visualization.png)
![Curvas ROC](assets/images/roc_curves.png)
```

## 📊 Estructura Recomendada

```
docs/
├── _config.yml
├── index.md
├── README.md (este archivo)
└── assets/
    └── images/
        ├── pca_2d_visualization.png
        ├── roc_curves.png
        ├── confusion_matrix_rf.png
        └── feature_importance.png
```

## 🔧 Desarrollo Local (Opcional)

Para previsualizar el sitio localmente antes de publicar:

### Instalar Jekyll

```bash
# macOS
gem install bundler jekyll

# Crear Gemfile
cat > Gemfile << EOF
source "https://rubygems.org"
gem "github-pages", group: :jekyll_plugins
gem "webrick"
EOF

bundle install
```

### Ejecutar Servidor Local

```bash
cd docs
bundle exec jekyll serve
```

Visita: `http://localhost:4000`

## ✅ Checklist de Publicación

- [ ] Verificar que todas las secciones del `index.md` están completas
- [ ] Copiar imágenes de resultados a `docs/assets/images/`
- [ ] Actualizar rutas de imágenes en `index.md`
- [ ] Verificar que las referencias están completas y formateadas
- [ ] Configurar GitHub Pages en Settings
- [ ] Esperar 1-2 minutos para despliegue
- [ ] Visitar URL pública y verificar que todo se visualiza correctamente
- [ ] Compartir URL con el equipo y profesor

## 🎯 Tips

- **Markdown Preview**: Usa la extensión de VS Code para previsualizar antes de subir
- **Cambios Incrementales**: Haz commits pequeños y verifica en la URL pública
- **Cache de Navegador**: Si no ves cambios, prueba con Ctrl+Shift+R (hard refresh)
- **Errores de Build**: Revisa la pestaña "Actions" en GitHub para ver logs de Jekyll

---

**Última actualización**: Diciembre 2025
