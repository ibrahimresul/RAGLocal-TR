NLTK_TURKISH_STOPWORDS = frozenset({
    "acaba", "ama", "aslında", "az", "bazı", "belki", "biri", "birkaç",
    "birşey", "biz", "bu", "çok", "çünkü", "da", "daha", "de", "defa",
    "diye", "eğer", "en", "gibi", "hem", "hep", "hepsi", "her", "hiç",
    "için", "ile", "ise", "kez", "ki", "kim", "mı", "mu", "mü", "nasıl",
    "ne", "neden", "nerde", "nerede", "nereye", "niçin", "niye", "o",
    "sanki", "şey", "siz", "şu", "tüm", "ve", "veya", "ya", "yani",
})

QUERY_NOISE_TERMS = frozenset({
    "alınmalıdır", "anlaşılabilir", "anlaşılır", "arasında", "arasındaki",
    "bir", "eder", "edilmelidir", "edilir", "fark", "farkı", "farkları",
    "gerekir", "gerekli", "gereklidir", "hangi", "izlenir", "kaç",
    "kullanılmalıdır", "kullanılır", "midir", "mi", "nedir", "nelerdir",
    "olan", "olarak", "olmalı", "olmalıdır", "olur", "önemli", "önemlidir",
    "önlenebilir", "önlenir", "önerir", "seçilir", "seçilmelidir",
    "tutulmalıdır", "uygulanmalıdır", "var", "yapılandırılmalıdır", "yapılır",
    "yazılmalıdır", "yok", "çalışır",
})

QUESTION_STOPWORDS = NLTK_TURKISH_STOPWORDS | QUERY_NOISE_TERMS
