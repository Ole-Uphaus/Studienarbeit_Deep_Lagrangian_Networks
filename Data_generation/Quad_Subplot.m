% -------------------------------------------------------------
% Autor:      Ole Uphaus
% Datum:      24.07.2025
% Beschreibung:
% Dies ist eine Matlab funktion für einen einfachen Subplot mit 2 Plots für die
% Studienarbeit.
% -------------------------------------------------------------

function Quad_Subplot(x, y_cell, xlabel_str, ylabel_str_cell, title_str_cell, legend_label_cell, filename, save_pdf, print_legend)

    % Textbreite Latex in cm (um figure genauso groß zu machen - in latex dann
    % skalierung == 1)
    textwidth_cm = 13.75;

    % Größe der Ausgabe definieren (Breite x Höhe in cm)
    fig_width = textwidth_cm;
    fig_height = 8.5;

    % Figure mit definierter Größe erzeugen
    figure('Visible', 'on', ...
           'Units', 'centimeters', ...
           'Position', [2 2 fig_width fig_height], ...
           'Color', 'w');

    % Subplot erstellen
    for i = 1:4
        subplot(2,2,i);

        % Mehrere Signale plotten (jede Spalte ist ein Signal)
        hold on;
        [~, n_signals] = size(y_cell{i});   % Anzahl der Signale
        for k = 1:n_signals
            plot(x, y_cell{i}(:, k), 'LineWidth', 1.5);
        end
        hold off;
        box on;
        grid on;
    
        % Achsenbereich einstellen
        yl = ylim();
        dy = diff(yl);
        ylim([yl(1) - 0.05*dy, yl(2) + 0.05*dy]);
    
        % Alles weiß 
        set(gca, 'Color', 'w'); 
    
        % Erst allgemeine Schriftgröße setzen
        set(gca, 'FontName', 'Latin Modern Roman');
        set(gca, 'FontSize', 11, 'TickLabelInterpreter', 'latex');
        
        % Beschriftungen
        if i == 3 || i == 4
            xlabel(xlabel_str, 'Interpreter', 'latex', 'FontSize', 11);
        end
        
        ylabel(ylabel_str_cell{i}, 'Interpreter', 'latex', 'FontSize', 11);
        title(title_str_cell{i}, 'Interpreter', 'latex', 'FontSize', 12);
        if print_legend
            legend(legend_label_cell{i}, 'Interpreter', 'latex', 'FontSize', 11, 'Location','best');
        end
    
        % Rand festlegen
        % set(gca, 'Position', [0.20, 0.17, 0.68, 0.72]); % Empfehlung [0.15, 0.17, 0.78, 0.72]
    end
    
    % Figure füllt gesamten Plot aus
    set(gcf, 'PaperUnits', 'centimeters');
    set(gcf, 'PaperSize', [fig_width fig_height]);
    set(gcf, 'PaperPosition', [0 0 fig_width fig_height]);
        
    if save_pdf
        print(gcf, filename, '-dpdf');
    end

end


