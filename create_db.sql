CREATE TABLE IF NOT EXISTS sintomas (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    sintoma TEXT NOT NULL,
    parte_corpo TEXT NOT NULL,
    intensidade INTEGER NOT NULL,
    diagnostico TEXT NOT NULL
);
