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
        other_symptoms: formData.get('other_symptoms') || '',
        fever: formData.get('fever') ? parseInt(formData.get('fever')) : 0,
        headache: formData.get('headache') ? parseInt(formData.get('headache')) : 0,
        eye_pain: formData.get('eye_pain') ? parseInt(formData.get('eye_pain')) : 0,
        joint_muscle_pain: formData.get('joint_muscle_pain') ? parseInt(formData.get('joint_muscle_pain')) : 0,
        nausea_vomiting: formData.get('nausea_vomiting') ? parseInt(formData.get('nausea_vomiting')) : 0,
        rash: formData.get('rash') ? parseInt(formData.get('rash')) : 0,
        bleeding: formData.get('bleeding') ? parseInt(formData.get('bleeding')) : 0
      };

      // Loading state
      submitBtn.disabled = true;
      btnText.textContent = 'جاري التحليل...';
      btnIcon.style.display = 'none';
      spinner.style.display = 'inline-block';
      errorMsg.textContent = '';

      try {
        // Use localhost URL for local dev, else use relative path for Render
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
        errorMsg.textContent = 'تعذر الاتصال بالخادم. تأكد من إدخال البيانات الصحيحة أو الاتصال بالإنترنت.';
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
    } else if (data.severity_level === 'Medium') {
      resultTitle.textContent = 'اشتباه متوسط للإصابة';
      resultTitle.style.color = 'var(--warning)';
      iconElement.className = 'ph-bold ph-warning';
    } else {
      resultTitle.textContent = 'احتمالية ضعيفة للإصابة';
      resultTitle.style.color = 'var(--success)';
      iconElement.className = 'ph-bold ph-check-circle';
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
        <div>حمى الضنك: <strong>${data.disease_breakdown.dengue}%</strong></div>
        <div>الملاريا: <strong>${data.disease_breakdown.malaria}%</strong></div>
        <div>طبيعي/أخرى: <strong>${data.disease_breakdown.normal}%</strong></div>
      `;
    }

    personalAdviceContainer.style.display = 'block';
    personalAdviceText.innerHTML = `<strong>بناءً على نتيجتك (${data.probability}%):</strong> ${data.medical_advice}`;
  }
});
