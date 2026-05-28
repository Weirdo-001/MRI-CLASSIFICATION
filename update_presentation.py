import os
import re

with open('presentation.html', 'r', encoding='utf-8') as f:
    content = f.read()

# 1. Update Title Subtitle
content = content.replace(
    '<div class=\"title-highlight\">AWS EC2 • Streamlit • TensorFlow • Production Pipeline</div>',
    '<div class=\"title-highlight\">AWS EC2 & Streamlit Cloud • TensorFlow & Grad-CAM • PySpark • DevOps</div>'
)

# 2. Add Architecture Diagram Slide after Slide 3
arch_slide = '''
        <!-- SLIDE 3.5: ARCHITECTURE DIAGRAM -->
        <div class="slide">
            <div class="slide-content">
                <div class="container">
                    <h2 class="slide-title">🗺️ System Architecture Diagram</h2>
                    <div style="width: 100%; height: 60vh; background: rgba(20, 33, 63, 0.6); border: 2px dashed var(--accent); border-radius: 12px; display: flex; flex-direction: column; justify-content: center; align-items: center; margin-top: 20px;">
                        <div style="font-size: 48px; margin-bottom: 15px;">🖼️</div>
                        <div style="color: var(--text-secondary); font-size: 16px;">[ Insert High-Level Architecture Diagram Here ]</div>
                        <div style="color: var(--text-muted); font-size: 12px; margin-top: 10px;">Suggested: Flow from User -> Streamlit Cloud / EC2 -> TF Model (Grad-CAM) -> Output</div>
                    </div>
                </div>
            </div>
        </div>
'''
content = content.replace('<!-- SLIDE 4: DEPLOYMENT STEPS - PARTS 1-3 -->', arch_slide + '\n        <!-- SLIDE 4: DEPLOYMENT STEPS - PARTS 1-3 -->')

# 3. Add PySpark and Grad-CAM slides after ML Model (Slide 7)
pyspark_gradcam_slides = '''
        <!-- SLIDE 7.1: EXPLAINABLE AI (GRAD-CAM) -->
        <div class="slide">
            <div class="slide-content">
                <div class="container">
                    <h2 class="slide-title">🔥 Explainable AI: Grad-CAM</h2>
                    <div class="grid-2">
                        <div class="card">
                            <div class="card-icon">🔍</div>
                            <div class="card-title">Why Grad-CAM?</div>
                            <div class="card-text">
                                Medical AI needs transparency. Grad-CAM (Gradient-weighted Class Activation Mapping) highlights the exact regions in the MRI that led to the model\\'s prediction, building trust with healthcare professionals.
                            </div>
                        </div>
                        <div class="card">
                            <div class="card-icon">⚙️</div>
                            <div class="card-title">Implementation</div>
                            <div class="card-text">
                                Extracts gradients from the <strong>top_conv</strong> layer of EfficientNet-B3. Applies power-normalization to emphasize the hottest activation zones and overlays a Jet colormap on the original MRI.
                            </div>
                        </div>
                    </div>
                </div>
            </div>
        </div>

        <!-- SLIDE 7.2: PYSPARK DATA PIPELINE -->
        <div class="slide">
            <div class="slide-content">
                <div class="container">
                    <h2 class="slide-title">⚡ Big Data Processing: PySpark</h2>
                    <div class="grid-2">
                        <div class="card">
                            <div class="card-icon">📈</div>
                            <div class="card-title">Data Inventory & Analytics</div>
                            <div class="card-text">
                                Built a PySpark pipeline for handling large-scale medical datasets. Processes image metadata and performs exploratory data analysis (class distribution, file validation) efficiently using distributed computing principles.
                            </div>
                        </div>
                        <div class="card">
                            <div class="card-icon">🛠️</div>
                            <div class="card-title">Technical Stack</div>
                            <div class="card-text">
                                Uses <strong>Apache Spark (PySpark)</strong> with Java (OpenJDK) backend on EC2. Designed to scale seamlessly if the dataset grows from thousands of MRIs to millions.
                            </div>
                        </div>
                    </div>
                </div>
            </div>
        </div>
'''
content = content.replace('<!-- SLIDE 8: FEATURES -->', pyspark_gradcam_slides + '\n        <!-- SLIDE 8: FEATURES -->')

# 4. Update total slides in JS and HTML
content = content.replace('const totalSlides = 9;', 'const totalSlides = 12;')
content = content.replace('<span id="totalSlides">9</span>', '<span id="totalSlides">12</span>')

# 5. Add Streamlit Cloud to Deployment / Conclusion
content = content.replace('deployed on <strong>AWS EC2</strong>', 'deployed on <strong>AWS EC2</strong> and <strong>Streamlit Cloud</strong>')
content = content.replace('<strong>AWS EC2 Deployed</strong> on t3.medium instance', '<strong>Dual Deployment</strong> on AWS EC2 & Streamlit Cloud')

with open('presentation_v2.html', 'w', encoding='utf-8') as f:
    f.write(content)

print("Created presentation_v2.html successfully.")
