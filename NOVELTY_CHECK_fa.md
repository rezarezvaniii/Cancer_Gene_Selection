# بررسی نوآوری روش پیشنهادی (kSV-RIME-LF) — برای استفاده در بخش Related Work / توجیه نوآوری مقاله

این فایل نتیجه‌ی یک بررسی واقعی وب (نه از حافظه) درباره‌ی این سؤاله: «آیا ترکیب
فیلتر SHAP/kSV + الگوریتم RIME (با هر استراتژی بهبودی) قبلاً جایی منتشر شده؟»
تاریخ بررسی: اوت ۲۰۲۶.

## نتیجه‌ی خلاصه

| جزء | آیا به‌تنهایی قبلاً وجود داشته؟ | منبع |
|---|---|---|
| فیلتر SHAP/kSV برای رتبه‌بندی ژن | بله (روش عمومی شناخته‌شده) | متعدد |
| الگوریتم RIME (پایه) | بله | Su et al., 2023, Neurocomputing |
| Lévy Flight (به‌عنوان استراتژی بهبود متاهیوریستیک) | بله (روی ده‌ها الگوریتم دیگه) | Yang & Deb, 2009 و مشتقاتش |
| **RIME + Lévy Flight (این ترکیب خاص)** | **پیدا نشد** | — |
| **SHAP/kSV filter + RIME (با هر wrapper)** | **پیدا نشد** | — |
| RIME + DLH (dimension learning-based hunting) روی microarray سرطان | **بله، قبلاً منتشر شده** | Cai et al., HDRIME، ۲۰۲۵/۲۰۲۶ |
| RIME + Soft-Besiege + Composite Mutation + Restart (اسم "IRIME") | **بله، قبلاً منتشر شده** | Huang et al., iScience، ۲۰۲۴ |

## نکات مهم برای متن مقاله

1. **اسم "IRIME" رو دیگه استفاده نکن.** از قبل توسط Huang, J., Chen, Y., Heidari,
   A. A., Liu, L., Chen, H., & Liang, G. (2024). "IRIME: Mitigating
   exploitation-exploration imbalance in RIME optimization for feature selection."
   *iScience*, 27(8), 110561. DOI: 10.1016/j.isci.2024.110561 گرفته شده (RIME +
   Soft Besiege + Composite Mutation + Restart، روی UCI/SBCB با KNN، نه microarray
   سرطان با SVM).

2. **اگه استراتژی DLH رو به RIME اضافه کنی، دیگه ادعای نوآوری مستقل نداری** — چون
   Cai, Z., Chen, T., Cai, Z., Chen, H., & Yu, S. (2026). "HDRIME: An improved RIME
   algorithm for global search and feature selection on gene expression microarray
   data." *Biomedical Signal Processing and Control*, 113(Part A), 108989. DOI:
   10.1016/j.bspc.2025.108989 دقیقاً همین کار رو کرده: چکیده‌ی رسمی مقاله عیناً از
   "the dimension learning-based hunting strategy" به‌عنوان یکی از چهار مکانیزم اصلی
   HDRIME اسم می‌بره (در کنار clustering+Cauchy mutation initialization، جهش/تقاطع
   DE/rand/1، و reflective boundary control) — روی داده‌ی microarray سرطان، دقیقاً
   دامنه‌ی همین پروژه. کد این مقاله هم روی گیت‌هاب موجوده:
   https://github.com/AlexandreShogenji/HDRIME

3. **چیزی که واقعاً می‌تونی ادعا کنی جدیده:** ترکیب فیلتر SHAP/kSV (که چارچوبش رو
   از مقاله‌ی پایه‌ی kSV-IGWO — Afreen et al., 2025, Knowledge and Information
   Systems — گرفتی) با هسته‌ی RIME + Lévy Flight به‌عنوان wrapper. این ترکیب خاص در
   هیچ جست‌وجویی پیدا نشد. روایت پیشنهادی برای بخش Introduction/Related Work مقاله:

   > «چارچوب فیلتر+wrapper مقاله‌ی kSV-IGWO (Afreen et al., 2025) را حفظ کردیم، اما
   > wrapper آن (GWO+DLH) را با یک نسخه‌ی بهبودیافته از RIME (Su et al., 2023) که با
   > استراتژی Lévy Flight (Yang & Deb, 2009) تقویت شده جایگزین کردیم. برخلاف
   > IRIME (Huang et al., 2024) و HDRIME (Cai et al., 2025) که RIME را به‌ترتیب با
   > Soft-Besiege/Composite-Mutation/Restart و با DLH ترکیب کرده‌اند، این پژوهش برای
   > نخستین‌بار RIME را با Lévy Flight ترکیب کرده و آن را — در کنار یک فیلتر
   > مبتنی‌بر Shapley value — روی داده‌های microarray سرطان با طبقه‌بند SVM ارزیابی
   > می‌کند.»

4. **هنوز باید مراقب باشی:** این یک بررسی با جست‌وجوی وب است، نه یک بررسی سیستماتیک
   کامل (systematic literature review). قبل از ارسال نهایی مقاله، حتماً یک بار دیگه
   (ترجیحاً با کمک استاد راهنما یا دیتابیس‌های تخصصی مثل Scopus/Web of Science) این
   ترکیب رو چک کن — مخصوصاً چون RIME الگوریتم نسبتاً جدیدیه (۲۰۲۳) و مقاله‌های زیادی
   دارن هرماه روش‌های بهبود جدید براش منتشر می‌کنن.
