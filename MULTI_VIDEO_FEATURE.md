# 🎬 Multi-Video Upload Feature - DANCEPIXEL

## ✨ Neue Features

### 1. **Mehrere Videos gleichzeitig hochladen**
- Wähle mehrere Videodateien auf einmal aus (Strg+Klick oder Drag & Drop)
- Alle Videos werden automatisch in die Warteschlange gestellt

### 2. **Parallele Verarbeitung**
- Alle Videos werden gleichzeitig auf dem Server verarbeitet
- Jeder User hat seine eigene Session mit separater Video-Liste

### 3. **Video-Queue mit Live-Status**
- Übersicht aller hochgeladenen Videos mit Status
- Status-Anzeigen:
  - **Queued** (Gelb): Video wartet auf Verarbeitung
  - **Processing X%** (Gelb): Video wird verarbeitet mit Fortschritt
  - **Completed** (Grün): Video fertig verarbeitet
  - **Failed** (Rot): Fehler bei Verarbeitung

### 4. **Dynamische Video-Auswahl**
- Klicke auf ein fertiges Video in der Liste
- Das Interface lädt automatisch die Pose-Daten
- Wechsle zwischen verschiedenen verarbeiteten Videos

## 🚀 Verwendung

### Frontend (DANCEPIXEL.html)
```
1. Öffne DANCEPIXEL.html im Browser
2. Klicke auf "Select Video File(s)" oder Drag & Drop mehrere Videos
3. Videos werden automatisch hochgeladen und verarbeitet
4. Warte bis Status "Completed" zeigt
5. Klicke auf ein fertiges Video → Interface lädt es automatisch
6. Wechsle zwischen Videos durch Klick in der Liste
```

### Backend (pose_server_ngrok.py)

```bash
# Server starten
python pose_server_ngrok.py
```

## 📡 API Endpunkte

### POST `/upload`
```
Upload ein Video und stelle es in die Queue
Params:
  - file: Video-Datei
  - user_session: (optional) Session ID

Returns:
  - job_id: ID für Status-Polling
  - user_session: Session ID
```

### GET `/job-status/{job_id}`
```
Hole Status eines Jobs
Returns:
  - status: queued | processing | completed | failed
  - progress: 0-100
  - result: Pose-Daten (nur wenn completed)
```

### GET `/user-jobs/{user_session}`
```
Hole alle Jobs eines Users
Returns:
  - Liste aller Jobs mit Status und Metadaten
```

## 🔧 Technische Details

### Backend Architektur
- **Async Processing**: FastAPI + asyncio für nicht-blockierende Verarbeitung
- **Job Queue**: In-Memory Storage für Video-Jobs
- **User Sessions**: Jeder User hat eigene Session-ID
- **Status Polling**: Frontend pollt alle 2 Sekunden für Updates

### Frontend State Management
```javascript
State.userSession       // Eindeutige User-Session ID
State.videoJobs[]       // Liste aller Video-Jobs
State.selectedJobId     // Aktuell ausgewähltes Video
State.pollingInterval   // Intervall für Status-Updates
```

### Multi-User Support
- Jeder User bekommt eine eindeutige Session-ID
- Videos werden pro Session gruppiert
- Keine Kollisionen zwischen verschiedenen Users

## 🎨 UI Komponenten

### Video Queue Sidebar
```html
<div id="videoQueueSection">
  <div class="video-queue">
    <!-- Video Items mit Status -->
  </div>
</div>
```

### Video Item States
- `.video-item.processing` - Gelber Hintergrund
- `.video-item.completed` - Grüner Hintergrund
- `.video-item.failed` - Roter Hintergrund
- `.video-item.selected` - Schwarzer Hintergrund (ausgewählt)

## 📝 Code-Beispiele

### Video hochladen (JavaScript)
```javascript
// Mehrere Videos hochladen
await VideoModule.uploadMultipleVideos(filesArray);

// Einzelnes Video hochladen
await VideoModule.uploadSingleVideo(file);
```

### Video aus Liste laden
```javascript
// Automatisch wenn User auf fertiges Video klickt
VideoQueueModule.loadVideo(job);
```

### Job-Status prüfen
```javascript
// Automatisches Polling
VideoQueueModule.startPolling();

// Manuell Status abrufen
await VideoQueueModule.updateJobStatuses();
```

## 🐛 Troubleshooting

### Videos werden nicht verarbeitet
- Prüfe ob Server läuft: `http://localhost:5000/info`
- Schaue in Browser Console nach Fehlern
- Prüfe Server-Logs für Processing-Errors

### Status wird nicht aktualisiert
- Polling läuft automatisch nach Upload
- Manuell starten: `VideoQueueModule.startPolling()`
- Polling stoppt automatisch wenn alle Jobs fertig

### Video lässt sich nicht laden
- Stelle sicher dass Job Status = "completed"
- Prüfe ob Video-Datei noch im Browser verfügbar ist
- Neu-Upload wenn nötig

## ⚡ Performance-Tipps

1. **Kleine Videos zuerst**: Kurze Videos werden schneller verarbeitet
2. **Server-Hardware**: GPU empfohlen für schnellere Pose Detection
3. **Batch-Size**: Bei vielen Videos, in Gruppen hochladen
4. **Cleanup**: Alte Jobs werden nach 24h automatisch gelöscht (TASK_TTL)

## 🎯 Zukünftige Erweiterungen

- [ ] Redis/Celery für echte Worker-Pools
- [ ] Video-Thumbnails in Queue
- [ ] Download verarbeiteter Videos
- [ ] Job-History speichern
- [ ] Prioritäts-Queue
- [ ] WebSocket für Echtzeit-Updates

---

**Version:** 2.0.0  
**Datum:** 13. Januar 2026  
**Autor:** GitHub Copilot für Emil
