document.addEventListener('DOMContentLoaded', () => {
    // --- Routing System ---
    const navLinks = document.querySelectorAll('.nav-link');
    const pageSections = document.querySelectorAll('.page-section');

    // Function to navigate between pages
    window.navigate = function (pageId) {
        // Update URL hash without jumping
        history.pushState(null, null, `#${pageId}`);

        // Hide all sections
        pageSections.forEach(section => {
            section.classList.remove('active-page');
            section.style.display = 'none'; // Force hide to avoid visual glitches
        });

        // Remove active class from all links
        navLinks.forEach(link => link.classList.remove('active'));

        // Show target section
        const targetSection = document.getElementById(pageId);
        if (targetSection) {
            targetSection.style.display = 'block';
            // Small timeout to allow display:block to apply before adding class for animation
            setTimeout(() => {
                targetSection.classList.add('active-page');
            }, 10);
        }

        // Highlight active link
        const targetLink = document.querySelector(`.nav-link[data-page="${pageId}"]`);
        if (targetLink) {
            targetLink.classList.add('active');
        }

        // Scroll to top smoothly
        window.scrollTo({ top: 0, behavior: 'smooth' });
    };

    // Listen to hash changes (browser back/forward buttons)
    window.addEventListener('hashchange', () => {
        const hash = window.location.hash.substring(1) || 'landing';
        navigate(hash);
    });

    // Initial load handling
    const initialHash = window.location.hash.substring(1) || 'landing';
    navigate(initialHash);

    // Intercept nav clicks
    navLinks.forEach(link => {
        link.addEventListener('click', (e) => {
            e.preventDefault();
            const page = link.getAttribute('data-page');
            navigate(page);
        });
    });

    // --- Mobile Menu ---
    const mobileBtn = document.querySelector('.mobile-menu-btn');
    const navLinksContainer = document.querySelector('.nav-links');

    mobileBtn.addEventListener('click', () => {
        const isDisplayed = navLinksContainer.style.display === 'flex';
        if (isDisplayed) {
            navLinksContainer.style.display = 'none';
        } else {
            navLinksContainer.style.display = 'flex';
            navLinksContainer.style.flexDirection = 'column';
            navLinksContainer.style.position = 'absolute';
            navLinksContainer.style.top = '100%';
            navLinksContainer.style.left = '0';
            navLinksContainer.style.width = '100%';
            navLinksContainer.style.background = 'white';
            navLinksContainer.style.padding = '1rem';
            navLinksContainer.style.boxShadow = '0 10px 15px rgba(0,0,0,0.05)';
        }
    });

    // --- Form Handling and API Integration ---
    const diagnosisForm = document.getElementById('diagnosisForm');
    const submitBtn = document.getElementById('submitBtn');
    const spinner = document.querySelector('.loader-spinner');
    const btnText = submitBtn.querySelector('span');
    const btnIcon = submitBtn.querySelector('i');
    const errorMsg = document.getElementById('error-message');

    // Result DOM Elements
    const resultCard = document.getElementById('resultCard');
    const resultIcon = document.getElementById('resultIcon');
    const resultTitle = document.getElementById('resultTitle');
    const progressFill = document.getElementById('probabilityFill');
    const progressText = document.getElementById('probabilityText');
    const resultMessage = document.getElementById('resultMessage');
    const resultActions = document.getElementById('resultActions');

    // Advice DOM Elements
    const personalAdviceContainer = document.getElementById('personalAdviceContainer');
    const personalAdviceText = document.getElementById('personalAdviceText');

    if (diagnosisForm) {
        diagnosisForm.addEventListener('submit', async (e) => {
            e.preventDefault();

            // Collect Form Data
            const formData = new FormData(diagnosisForm);
            const features = {
                patient_name: formData.get('patient_name') || '',
                age: formData.get('age') ? parseInt(formData.get('age')) : 30,
                gender: formData.get('gender') ? parseInt(formData.get('gender')) : 0,
                other_symptoms: formData.get('other_symptoms') || '',
                fever: formData.get('fever') ? parseInt(formData.get('fever')) : 0,
                headache: formData.get('headache') ? parseInt(formData.get('headache')) : 0,
                eye_pain: formData.get('eye_pain') ? parseInt(formData.get('eye_pain')) : 0,
                joint_muscle_pain: formData.get('joint_muscle_pain') ? parseInt(formData.get('joint_muscle_pain')) : 0,
                nausea_vomiting: formData.get('nausea_vomiting') ? parseInt(formData.get('nausea_vomiting')) : 0,
                rash: formData.get('rash') ? parseInt(formData.get('rash')) : 0,
                bleeding: formData.get('bleeding') ? parseInt(formData.get('bleeding')) : 0
            };

            // Show Loading State
            submitBtn.disabled = true;
            btnText.textContent = 'جاري التحليل...';
            btnIcon.style.display = 'none';
            spinner.style.display = 'inline-block';
            errorMsg.textContent = '';

            try {
                // Determine API URL
                const apiUrl = '/api/predict';

                const response = await fetch(apiUrl, {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify(features)
                });

                if (!response.ok) {
                    throw new Error('حدث خطأ أثناء الاتصال بالخادم.');
                }

                const data = await response.json();

                if (data.error) {
                    throw new Error(data.error);
                }

                // Process Result and Update UI
                updateResultUI(data);

                // Navigate to Result Page
                navigate('result');

            } catch (error) {
                console.error('Prediction Error:', error);
                errorMsg.textContent = 'تعذر الاتصال بخادم الذكاء الاصطناعي. يرجى التأكد من تشغيل الخادم (Backend).';
            } finally {
                // Restore Button State
                submitBtn.disabled = false;
                btnText.textContent = 'تحليل النتيجة';
                btnIcon.style.display = 'inline-block';
                spinner.style.display = 'none';
            }
        });
    }

    function updateResultUI(data) {
        resultCard.style.display = 'block';
        resultActions.style.display = 'flex';

        // Update Title & Icon
        const isDengue = data.prediction === 1;
        const iconElement = resultIcon.querySelector('i');

        resultIcon.className = 'result-icon pulse-animation'; // reset

        if (data.severity_level === 'High') {
            resultTitle.textContent = 'احتمالية عالية للإصابة';
            resultTitle.style.color = 'var(--danger)';
            resultIcon.classList.add('high');
            iconElement.className = 'ph-bold ph-warning-octagon';
        } else if (data.severity_level === 'Medium') {
            resultTitle.textContent = 'اشتباه متوسط للإصابة';
            resultTitle.style.color = 'var(--warning)';
            resultIcon.classList.add('medium');
            iconElement.className = 'ph-bold ph-warning';
        } else {
            resultTitle.textContent = 'احتمالية ضعيفة للإصابة';
            resultTitle.style.color = 'var(--success)';
            resultIcon.classList.add('low');
            iconElement.className = 'ph-bold ph-check-circle';
        }

        // Update Progress Bar
        // We delay the width update slightly so the CSS transition triggers
        setTimeout(() => {
            progressFill.style.width = `${data.probability}%`;

            // Adjust bar color based on probability
            if (data.probability > 75) progressFill.style.background = 'var(--danger)';
            else if (data.probability > 40) progressFill.style.background = 'var(--warning)';
            else progressFill.style.background = 'var(--success)';
        }, 300);

        progressText.textContent = `${data.probability}%`;

        // Update Details Message
        resultMessage.innerHTML = `<strong>التحليل الآلي:</strong> ${data.message}`;

        // Multi Disease Breakdown
        const multiDiseaseCard = document.getElementById('multiDiseaseCard');
        const diseaseBreakdown = document.getElementById('diseaseBreakdown');

        if (data.disease_breakdown && multiDiseaseCard && diseaseBreakdown) {
            multiDiseaseCard.style.display = 'block';
            diseaseBreakdown.innerHTML = `
                <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 0.5rem;">
                    <span>حمى الضنك</span>
                    <strong>${data.disease_breakdown.dengue}%</strong>
                </div>
                <div class="progress-bar" style="height: 8px; margin-bottom: 1rem;"><div class="progress-fill" style="width: ${data.disease_breakdown.dengue}%; background: var(--danger);"></div></div>
                
                <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 0.5rem;">
                    <span>الملاريا</span>
                    <strong>${data.disease_breakdown.malaria}%</strong>
                </div>
                <div class="progress-bar" style="height: 8px; margin-bottom: 1rem;"><div class="progress-fill" style="width: ${data.disease_breakdown.malaria}%; background: var(--warning);"></div></div>
                
                <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 0.5rem;">
                    <span>أعراض طبيعية / أخرى</span>
                    <strong>${data.disease_breakdown.normal}%</strong>
                </div>
                <div class="progress-bar" style="height: 8px;"><div class="progress-fill" style="width: ${data.disease_breakdown.normal}%; background: var(--success);"></div></div>
            `;
        }

        // Update Advice Page with Custom Advice
        personalAdviceContainer.style.display = 'block';
        personalAdviceText.innerHTML = `<strong>بناءً على نتيجتك (${data.probability}%):</strong> ${data.medical_advice}`;
    }
});
