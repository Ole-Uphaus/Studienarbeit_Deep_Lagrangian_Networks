% -------------------------------------------------------------
% Autor:      Ole Uphaus
% Datum:      24.07.2025
% Beschreibung:
% Dies ist eine Matlab funktion für einen einfachen Plot für die
% Studienarbeit.
% -------------------------------------------------------------

function Single_Plot(x, y, xlabel_str, ylabel_str, title_str, legend_label, filename, save_pdf, print_legend)

    % Textbreite Latex in cm (um figure genauso groß zu machen - in latex dann
    % skalierung == 1)
    textwidth_cm = 13.75;

    % Größe der Ausgabe definieren (Breite x Höhe in cm)
    fig_width = textwidth_cm;
    fig_height = 8.25;

    % Figure mit definierter Größe erzeugen
    figure('Visible', 'on', ...
           'Units', 'centimeters', ...
           'Position', [2 2 fig_width fig_height], ...
           'Color', 'w');

    % Mehrere Signale plotten (jede Spalte ist ein Signal)
    hold on;
    [~, n_signals] = size(y);
    for k = 1:n_signals
        plot(x, y(:, k), 'LineWidth', 1.5);
    end
    hold off;
    box on;
    grid on;

    % Achsenbereich einstellen
    yl = ylim();
    dy = diff(yl);
    new_ylim = [yl(1) - 0.05*dy, yl(2) + 0.05*dy];
        ylim(new_ylim);

        % Nullinie etwas dicker machen
        if new_ylim(1) < 0 && new_ylim(2) > 0
            % neue Nullinie
            h0 = line(xlim(), [0 0], ...
                'Color', [0.8 0.8 0.8], ...
                'LineStyle', '-', ...
                'LineWidth', 0.7, ...
                'HandleVisibility', 'off');  % nicht in der Legende anzeigen

            % Achsenticks künstlich
            current_xlim = xlim();
            tick_offset = 0.01;

            tick_left = line([current_xlim(1), (current_xlim(2) - current_xlim(1))*tick_offset], [0 0], ...
                'Color', [0 0 0], ...
                'LineStyle', '-', ...
                'LineWidth', get(gca, 'LineWidth'), ...
                'HandleVisibility', 'off');  % nicht in der Legende anzeigen

            tick_right = line([(current_xlim(2) - current_xlim(1))*(1 - tick_offset), current_xlim(2)], [0 0], ...
                'Color', [0 0 0], ...
                'LineStyle', '-', ...
                'LineWidth', get(gca, 'LineWidth'), ...
                'HandleVisibility', 'off');  % nicht in der Legende anzeigen
        
            % Linie nach hinten schieben
            uistack(tick_left, 'bottom');
            uistack(tick_right, 'bottom');
            uistack(h0, 'bottom');
        end

    % Alles weiß 
    set(gca, 'Color', 'w'); 

    % Erst allgemeine Schriftgröße setzen
    set(gca, 'FontName', 'Latin Modern Roman');
    set(gca, 'FontSize', 11, 'TickLabelInterpreter', 'latex');
    
    % Beschriftungen
    xlabel(xlabel_str, 'Interpreter', 'latex', 'FontSize', 11);
    ylabel(ylabel_str, 'Interpreter', 'latex', 'FontSize', 11);
    title(title_str, 'Interpreter', 'latex', 'FontSize', 12);
    if print_legend
        legend(legend_label, 'Interpreter', 'latex', 'FontSize', 9, 'Location','best');
    end

    % Rand festlegen
    set(gca, 'Position', [0.20, 0.17, 0.68, 0.72]); % Empfehlung [0.15, 0.17, 0.78, 0.72]
    
    % Figure füllt gesamten Plot aus
    set(gcf, 'PaperUnits', 'centimeters');
    set(gcf, 'PaperSize', [fig_width fig_height]);
    set(gcf, 'PaperPosition', [0 0 fig_width fig_height]);
        
    if save_pdf
        print(gcf, filename, '-dpdf');
    end

end


