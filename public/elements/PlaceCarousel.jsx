import React, { useState } from "react";

export default function PlaceCarousel() {
    const places = props.places || [];
    
    const [isLocked, setIsLocked] = useState(false);
    const [selectedId, setSelectedId] = useState(null);

    const handleAddPlace = async (place) => {
        if (isLocked) return;

        setIsLocked(true);
        setSelectedId(place.id);

        console.log("➡️ Pulsante cliccato per:", place.title);

        try {
            // Cerca callAction nello scope globale di Chainlit o fallback su window
            const sendAction = typeof callAction !== "undefined" ? callAction : window.callAction;

            if (sendAction) {
                console.log("🚀 Invocazione action 'agent_add_place'...");
                
                await sendAction({
                    name: "agent_add_place",
                    payload: {
                        id: place.id,
                        title: place.title,
                        duration: place.visit_duration,
                        travel_time: place.travel_time,
                        travel_time_hours_to_final_destination: place.travel_time_hours_to_final_destination
                    }
                });

                console.log("✅ Action inviata a Chainlit con successo!");
            } else {
                console.error("❌ ERRORE: callAction non trovata nello scope!");
            }
        } catch (error) {
            console.error("❌ ERRORE durante callAction:", error);
            // Ripristina i pulsanti in caso di errore
            setIsLocked(false);
            setSelectedId(null);
        }
    };

    const handleInfoPlace = async (place) => {

        console.log("➡️ Pulsante cliccato per:", place.title);

        try {
            // Cerca callAction nello scope globale di Chainlit o fallback su window
            const sendAction = typeof callAction !== "undefined" ? callAction : window.callAction;

            if (sendAction) {
                console.log("🚀 Invocazione action 'agent_info_place'...");
                
                await sendAction({
                    name: "agent_info_place",
                    payload: {
                        id: place.id,
                    }
                });

                console.log("✅ Action inviata a Chainlit con successo!");
            } else {
                console.error("❌ ERRORE: callAction non trovata nello scope!");
            }
        } catch (error) {
            console.error("❌ ERRORE durante callAction:", error);
        }
    };

    return (
        <div
            style={{
                display: "flex",
                gap: "16px",
                overflowX: "auto",
                padding: "10px",
                width: "100%"
            }}
        >
            {places.map((place) => {
                const isThisSelected = selectedId === place.id;

                return (
                    <div
                        key={place.id}
                        style={{
                            minWidth: "280px",
                            borderRadius: "12px",
                            border: "1px solid #ddd",
                            padding: "12px",
                            flexShrink: 0,
                            opacity: isLocked && !isThisSelected ? 0.5 : 1,
                            transition: "all 0.3s ease"
                        }}
                    >
                        {place.image && (
                            <img
                                src={place.image}
                                style={{
                                    width: "100%",
                                    height: "160px",
                                    objectFit: "cover",
                                    borderRadius: "8px"
                                }}
                                alt={place.title || "Immagine luogo"}
                            />
                        )}

                        <h3 style={{ margin: "8px 0", fontWeight: "bold" }}>
                            {place.title}
                        </h3>

                        {/*togliere anche le graffe poi <p>🏷️ {place.category || "N/D"}</p>*/}

                        {place.distance != null && (
                            <p>📍 {place.distance} km dalla tappa precedente</p>
                        )}

                        {place.travel_time != null && (() => {
                            const rawValue = Number(place.travel_time);
                            const minutes = Math.round(rawValue * 60);
                            return <p>🚗 {minutes} min dalla tappa precedente</p>;
                        })()}

                        {place.travel_time_hours_to_final_destination != null && (() => {
                            const rawValue = Number(place.travel_time_hours_to_final_destination);
                            const minutes = Math.round(rawValue * 60);
                            return <p>🛏️ {minutes} min da alloggio</p>;
                        })()}
                        <p>⏱️ {place.visit_duration || "N/D"}</p>
                        {place.url && (
                                        <p>
                                            🔗{" "}
                                            <a
                                                href={place.url}
                                                target="_blank"
                                                rel="noopener noreferrer"
                                            >
                                                Sito web
                                            </a>
                                        </p>
                                    )}
                        
                        <button
                                onClick={() =>
                                    handleInfoPlace(place)
                                }
                                style={{
                                    flex: "0 0 42px",
                                    width: "100%",
                                    padding: "8px 0",
                                    borderRadius: "6px",
                                    backgroundColor: "#6B7280",
                                    color: "#fff",
                                    border: "none",
                                    cursor: "pointer",
                                    fontWeight: "600",
                                    fontSize: "16px"
                                }}
                                title="Informazioni"
                            >
                                ℹ️Info
                            </button>

                        <button
                            onClick={() => handleAddPlace(place)}
                            disabled={isLocked}
                            style={{
                                marginTop: "8px",
                                width: "100%",
                                padding: "8px 12px",
                                borderRadius: "6px",
                                backgroundColor: isThisSelected 
                                    ? "#10B981" 
                                    : (isLocked ? "#D1D5DB" : "#0070f3"),
                                color: isLocked && !isThisSelected ? "#6B7280" : "#fff",
                                border: "none",
                                cursor: isLocked ? "not-allowed" : "pointer",
                                fontWeight: "600",
                                transition: "all 0.2s ease"
                            }}
                        >
                            {isThisSelected 
                                ? "✓ Selezionato" 
                                : (isLocked ? "Non disponibile" : "+ Aggiungi")}
                        </button>
                    </div>
                );
            })}
        </div>
    );
}