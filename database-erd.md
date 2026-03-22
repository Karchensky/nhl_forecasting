# Database ERD (SQLite)

Entity-relationship view of `data/nhl_forecasting.db` (SQLAlchemy models in `database/models.py`).

```mermaid
erDiagram
    teams ||--o{ players : "current_team_id"
    teams ||--o{ games : "home_team_id"
    teams ||--o{ games : "away_team_id"
    teams ||--o{ player_game_stats : "team_id"
    teams ||--o{ team_game_stats : "team_id"
    teams ||--o{ goalie_game_stats : "team_id"

    games ||--o{ player_game_stats : "game_id"
    games ||--o{ team_game_stats : "game_id"
    games ||--o{ goalie_game_stats : "game_id"
    games ||--o{ odds : "game_id"
    games ||--o{ model_outputs : "game_id"

    players ||--o{ player_game_stats : "player_id"
    players ||--o{ goalie_game_stats : "player_id"
    players ||--o{ odds : "player_id"
    players ||--o{ model_outputs : "player_id"

    teams {
        int team_id PK
        string abbreviation
        string full_name
        string conference
        string division
    }

    players {
        int player_id PK
        string full_name
        string position
        date birth_date
        int current_team_id FK
        boolean active
    }

    games {
        int game_id PK
        int season
        int game_type
        date game_date
        int home_team_id FK
        int away_team_id FK
        int home_score
        int away_score
        string game_state
        string venue
    }

    player_game_stats {
        int player_id PK_FK
        int game_id PK_FK
        int team_id FK
        int goals
        int shots
        int toi_seconds
        int pp_toi_seconds
        float corsi_for
        float fenwick_for
    }

    team_game_stats {
        int team_id PK_FK
        int game_id PK_FK
        int goals
        int shots
        boolean is_home
    }

    goalie_game_stats {
        int player_id PK_FK
        int game_id PK_FK
        int team_id FK
        int saves
        int goals_against
    }

    odds {
        int id PK
        int player_id FK
        int game_id FK
        string sportsbook
        string market
        int american_odds
        float implied_probability
        datetime retrieved_at
    }

    model_outputs {
        int id PK
        int player_id FK
        int game_id FK
        string model_version
        float predicted_probability
        datetime created_at
    }
```

## Notes

- **Composite keys:** `player_game_stats`, `team_game_stats`, and `goalie_game_stats` use `(player_id or team_id, game_id)` as the primary key.
- **Uniqueness:** `odds` is unique on `(player_id, game_id, sportsbook, market)`. `model_outputs` is unique on `(player_id, game_id, model_version)`.
- **Skater vs goalie:** Skater rows live in `player_game_stats`; goalies also have `goalie_game_stats`. The feature pipeline excludes goalies from skater `player_game_stats` when building the matrix.

Render this file in GitHub, GitLab, or any Mermaid-compatible viewer (VS Code extension “Markdown Preview Mermaid Support”, etc.).
