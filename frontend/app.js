document.addEventListener('DOMContentLoaded', () => {
  // --- Routing System ---
  const navLinks = document.querySelectorAll('.nav-link');
  const pageSections = document.querySelectorAll('.page-section');

  window.navigate = function (pageId) {
    history.pushState(null, null, `#${pageId}`);
    pageSections.forEach(section => {
      section.classList.remove('active-page');
      section.style.display = 'none';
    });
    navLinks.forEach(link => link.classList.remove('active'));

    const targetSection = document.getElementById(pageId);
    if (targetSection) {
      targetSection.style.display = 'block';
      setTimeout(() => targetSection.classList.add('active-page'), 10);
    }

    const targetLink = document.querySelector(`.nav-link[data-page="${pageId}"]`);
    if (targetLink) targetLink.classList.add('active');

    window.scrollTo({ top: 0, behavior: 'smooth' });
  };

  window.addEventListener('hashchange', () => {
    const hash = window.location.hash.substring(1) || 'landing';
    navigate(hash);
  });

  const initialHash = window.location.hash.substring(1) || 'landing';
  navigate(initialHash);

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

      const formData = new FormData(diagnosisForm);
      const features = {
        patient_name: formData.get('patient_name') || '',
        age: formData.get('age') ? parseInt(formData.get('age')) : 30,
        gender: formData.get('gender') ? parseInt(formData.get('gender')) : 0,
        fever: formData.get('fever') ? parseFloat(formData.get('fever')) : 0,
        headache: formData.get('headache') ? parseFloat(formData.get('headache')) : 0,
        fatigue: formData.get('fatigue') ? parseFloat(formData.get('fatigue')) : 0,
        vomiting: formData.get('vomiting') ? parseFloat(formData.get('vomiting')) : 0,
        eye_pain: formData.get('eye_pain') ? parseFloat(formData.get('eye_pain')) : 0,
        rash: formData.get('rash') ? parseFloat(formData.get('rash')) : 0,
        joint_pain: formData.get('joint_pain') ? parseFloat(formData.get('joint_pain')) : 0,
        bleeding: formData.get('bleeding') ? parseFloat(formData.get('bleeding')) : 0,
        chills: formData.get('chills') ? parseFloat(formData.get('chills')) : 0,
        sweating: formData.get('sweating') ? parseFloat(formData.get('sweating')) : 0,
        anemia: formData.get('anemia') ? parseFloat(formData.get('anemia')) : 0,
        jaundice: formData.get('jaundice') ? parseFloat(formData.get('jaundice')) : 0,
        abdominal_pain: formData.get('abdominal_pain') ? parseFloat(formData.get('abdominal_pain')) : 0,
        loss_of_appetite: formData.get('loss_of_appetite') ? parseFloat(formData.get('loss_of_appetite')) : 0,
        diarrhea_constipation: formData.get('diarrhea_constipation') ? parseFloat(formData.get('diarrhea_constipation')) : 0,
        other_symptoms: formData.get('other_symptoms') || ''
      };

      // Loading state
      submitBtn.disabled = true;
      btnText.textContent = 'جاري التحليل...';
      btnIcon.style.display = 'none';
      spinner.style.display = 'inline-block';
      errorMsg.textContent = '';

      try {
        const apiUrl = (window.location.hostname === 'localhost' || window.location.hostname === '127.0.0.1') 
          ? 'http://127.0.0.1:10000/api/predict' 
          : '/api/predict';
          
        const response = await fetch(apiUrl, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify(features)
        });

        if (!response.ok) throw new Error('خطأ في الاتصال بالخادم.');

        const data = await response.json();
        if (data.error) throw new Error(data.error);

        updateResultUI(data);
        navigate('result');
      } catch (error) {
        console.error('Prediction Error:', error);
        errorMsg.textContent = 'تعذر الاتصال بالخادم. تأكد من تشغيل السيرفر أو المحاولة لاحقاً.';
      } finally {
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

    const iconElement = resultIcon.querySelector('i');
    resultIcon.className = 'result-icon pulse-animation';

    if (data.severity_level === 'High') {
      resultTitle.textContent = 'احتمالية عالية للإصابة';
      resultTitle.style.color = 'var(--danger)';
      iconElement.className = 'ph-bold ph-warning-octagon';
      resultIcon.classList.add('high');
    } else if (data.severity_level === 'Medium') {
      resultTitle.textContent = 'اشتباه متوسط للإصابة';
      resultTitle.style.color = 'var(--warning)';
      iconElement.className = 'ph-bold ph-warning';
      resultIcon.classList.add('medium');
    } else {
      resultTitle.textContent = 'احتمالية ضعيفة للإصابة';
      resultTitle.style.color = 'var(--success)';
      iconElement.className = 'ph-bold ph-check-circle';
      resultIcon.classList.add('low');
    }

    setTimeout(() => {
      progressFill.style.width = `${data.probability}%`;
      if (data.probability > 75) progressFill.style.background = 'var(--danger)';
      else if (data.probability > 40) progressFill.style.background = 'var(--warning)';
      else progressFill.style.background = 'var(--success)';
    }, 300);

    progressText.textContent = `${data.probability}%`;
    resultMessage.innerHTML = `<strong>التحليل الآلي:</strong> ${data.message}`;

    const multiDiseaseCard = document.getElementById('multiDiseaseCard');
    const diseaseBreakdown = document.getElementById('diseaseBreakdown');
    if (data.disease_breakdown && multiDiseaseCard && diseaseBreakdown) {
      multiDiseaseCard.style.display = 'block';
      diseaseBreakdown.innerHTML = `
        <div class="disease-stat">
            <div class="stat-label">حمى الضنك:</div>
            <div class="stat-bar-container"><div class="stat-bar" style="width: ${data.disease_breakdown.dengue}%"></div></div>
            <div class="stat-percentage">${data.disease_breakdown.dengue}%</div>
        </div>
        <div class="disease-stat">
            <div class="stat-label">الملاريا:</div>
            <div class="stat-bar-container"><div class="stat-bar" style="width: ${data.disease_breakdown.malaria}%"></div></div>
            <div class="stat-percentage">${data.disease_breakdown.malaria}%</div>
        </div>
        <div class="disease-stat">
            <div class="stat-label">التايفويد:</div>
            <div class="stat-bar-container"><div class="stat-bar" style="width: ${data.disease_breakdown.typhoid}%"></div></div>
            <div class="stat-percentage">${data.disease_breakdown.typhoid}%</div>
        </div>
        <div class="disease-stat">
            <div class="stat-label">الإنفلونزا:</div>
            <div class="stat-bar-container"><div class="stat-bar" style="width: ${data.disease_breakdown.flu}%"></div></div>
            <div class="stat-percentage">${data.disease_breakdown.flu}%</div>
        </div>
        <div class="disease-stat">
            <div class="stat-label">كوفيد-19:</div>
            <div class="stat-bar-container"><div class="stat-bar" style="width: ${data.disease_breakdown.covid}%"></div></div>
            <div class="stat-percentage">${data.disease_breakdown.covid}%</div>
        </div>
      `;
    }

    personalAdviceContainer.style.display = 'block';
    personalAdviceText.innerHTML = `<strong>بناءً على نتيجتك (${data.probability}%):</strong> ${data.medical_advice}`;
  }
});
