var drawPayCode = function (qrcode, canvas, margin, logo, consumer = false) {
    var ctx = canvas.getContext('2d');
    var width = ctx.canvas.width;

    var cellCount = qrcode.getModuleCount();
    var cellRadius = ((width - (margin * 2)) / cellCount) / 2;
    var offset = cellRadius + margin

    var iconPercent = 32 / 260;
    var iconClipPercent = 35 / 260;

    var logoPercent = 67 / 260;
    var logoClipPercent = 72 / 260;

    // white background
    ctx.fillStyle = '#ffffff';
    ctx.fillRect(0, 0, width, width);

    if (consumer) {
        // PayMe red
        ctx.fillStyle = '#db0011';
    }
    else {
        // PM4B red
        ctx.fillStyle = '#c92a23';
    }

    // top left eye
    drawEye(ctx, width, margin, cellRadius, 0, 0, offset);
    // top right eye
    drawEye(ctx, width, margin, cellRadius, 2 * (cellCount - 7) * cellRadius, 0, offset);
    // bottom left eye
    drawEye(ctx, width, margin, cellRadius, 0, 2 * (cellCount - 7) * cellRadius, offset);

    var iconWidth = ((width - 2 * margin) * iconPercent)
    var logoWidth = ((width - 2 * margin) * logoPercent);

    var iconClip = width - cellRadius - margin - ((width - 2 * margin) * iconClipPercent)
    var logoLeftClip = (width / 2) - (((width - 2 * margin) * logoClipPercent) / 2);
    var logoRightClip = (width / 2) + (((width - 2 * margin) * logoClipPercent) / 2);

    // PayMe icon in bottom right
    drawIcon(ctx, iconWidth, width, margin, consumer);

    // business logo in the middle
    drawLogo(ctx, logo, logoWidth, width, margin, consumer);

    for (var r = 0; r < cellCount; r += 1) {
        for (var c = 0; c < cellCount; c += 1) {
            var x = c * cellRadius * 2 + offset;
            var y = r * cellRadius * 2 + offset;

            if (r < 7 && (c < 7 || c > cellCount - 8) || r > cellCount - 8 && c < 7) {
                // don't draw cells over the "eyes"
                continue;
            }
            else if (x >= iconClip && y >= iconClip) {
                // don't draw cells over the PM4B icon
                continue;
            }
            else if (x >= logoLeftClip && x < logoRightClip &&
                y >= logoLeftClip && y < logoRightClip) {
                // don't draw cells over the logo
                continue;
            }

            if (qrcode.isDark(r, c)) {
                ctx.beginPath();
                ctx.arc(x, y, cellRadius, 0, 2 * Math.PI);
                ctx.fill();
            }
        }
    }
}

var drawEye = function (ctx, width, margin, cellRadius, xOffset, yOffset, offset) {
    // outer edge
    ctx.beginPath();
    ctx.moveTo(xOffset + 11 * cellRadius + margin, yOffset + 0 * cellRadius + margin);
    ctx.arcTo(
        xOffset + 14 * cellRadius + margin, yOffset + 0 * cellRadius + margin,
        xOffset + 14 * cellRadius + margin, yOffset + 3 * cellRadius + margin,
        3 * cellRadius);

    ctx.arcTo(
        xOffset + 14 * cellRadius + margin, yOffset + 14 * cellRadius + margin,
        xOffset + 12 * cellRadius + margin, yOffset + 14 * cellRadius + margin,
        3 * cellRadius);

    ctx.arcTo(
        xOffset + 0 * cellRadius + margin, yOffset + 14 * cellRadius + margin,
        xOffset + 0 * cellRadius + margin, yOffset + 11 * cellRadius + margin,
        3 * cellRadius);

    ctx.arcTo(
        xOffset + 0 * cellRadius + margin, yOffset + 0 * cellRadius + margin,
        xOffset + 3 * cellRadius + margin, yOffset + 0 * cellRadius + margin,
        3 * cellRadius);
    ctx.lineTo(xOffset + 11 * cellRadius + margin, yOffset + 0 * cellRadius + margin);

    // inner edge
    ctx.moveTo(xOffset + 4 * cellRadius + margin, yOffset + 2 * cellRadius + margin);
    ctx.arcTo(
        xOffset + 2 * cellRadius + margin, yOffset + 2 * cellRadius + margin,
        xOffset + 2 * cellRadius + margin, yOffset + 4 * cellRadius + margin,
        2 * cellRadius);

    ctx.arcTo(
        xOffset + 2 * cellRadius + margin, yOffset + 12 * cellRadius + margin,
        xOffset + 4 * cellRadius + margin, yOffset + 12 * cellRadius + margin,
        2 * cellRadius);

    ctx.arcTo(
        xOffset + 12 * cellRadius + margin, yOffset + 12 * cellRadius + margin,
        xOffset + 12 * cellRadius + margin, yOffset + 10 * cellRadius + margin,
        2 * cellRadius);

    ctx.arcTo(
        xOffset + 12 * cellRadius + margin, yOffset + 2 * cellRadius + margin,
        xOffset + 10 * cellRadius + margin, yOffset + 2 * cellRadius + margin,
        2 * cellRadius);
    ctx.lineTo(xOffset + 4 * cellRadius + margin, yOffset + 2 * cellRadius + margin);

    ctx.closePath();
    ctx.fill();

    // central rect
    ctx.beginPath();
    ctx.moveTo(xOffset + 8 * cellRadius + margin, yOffset + 4 * cellRadius + margin);
    ctx.arcTo(
        xOffset + 10 * cellRadius + margin, yOffset + 4 * cellRadius + margin,
        xOffset + 10 * cellRadius + margin, yOffset + 6 * cellRadius + margin,
        cellRadius);

    ctx.arcTo(
        xOffset + 10 * cellRadius + margin, yOffset + 10 * cellRadius + margin,
        xOffset + 8 * cellRadius + margin, yOffset + 10 * cellRadius + margin,
        cellRadius);

    ctx.arcTo(
        xOffset + 4 * cellRadius + margin, yOffset + 10 * cellRadius + margin,
        xOffset + 4 * cellRadius + margin, yOffset + 8 * cellRadius + margin,
        cellRadius);

    ctx.arcTo(
        xOffset + 4 * cellRadius + margin, yOffset + 4 * cellRadius + margin,
        xOffset + 6 * cellRadius + margin, yOffset + 4 * cellRadius + margin,
        cellRadius);
    ctx.lineTo(xOffset + 8 * cellRadius + margin, yOffset + 4 * cellRadius + margin);
    ctx.closePath();
    ctx.fill();
}

var drawIcon = function (ctx, iconWidth, width, margin, consumer) {
    var icon = new Image();

    var xOffset = width - margin - iconWidth;
    var yOffset = width - margin - iconWidth;

    // Apple design guidelines state that corner radius is 80px for a 512px icon
    var cornerRadius = iconWidth * 80 / 512;
    var edgeLength = iconWidth - cornerRadius;

    icon.onload = function () {
        ctx.save();
        ctx.beginPath();

        ctx.moveTo(xOffset + edgeLength, yOffset);
        ctx.arcTo(
            xOffset + edgeLength + cornerRadius, yOffset,
            xOffset + edgeLength + cornerRadius, yOffset + cornerRadius,
            cornerRadius);

        ctx.arcTo(
            xOffset + edgeLength + cornerRadius, yOffset + edgeLength + cornerRadius,
            xOffset + edgeLength, yOffset + edgeLength + cornerRadius,
            cornerRadius);

        ctx.arcTo(
            xOffset, yOffset + edgeLength + cornerRadius,
            xOffset, yOffset + edgeLength,
            cornerRadius);

        ctx.arcTo(
            xOffset, yOffset,
            xOffset + cornerRadius, yOffset,
            cornerRadius);
        ctx.lineTo(xOffset + edgeLength, yOffset);
        ctx.closePath();

        ctx.clip();
        ctx.drawImage(icon, xOffset, yOffset, iconWidth, iconWidth);

        if (!consumer) {
            ctx.strokeStyle = "#767676";
            ctx.lineWidth = 0.5;
            ctx.stroke();
        }

        ctx.restore();
    };

    if (consumer) {
        icon.src = "data:image/png;base64,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"
    }
    else {
        icon.src = 'data:image/png;base64,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';
    }
}

var drawLogo = function (ctx, img, logoWidth, width, margin, consumer) {
    ctx.save();
    ctx.beginPath();
    ctx.arc(width / 2, width / 2, logoWidth / 2, 0, 2 * Math.PI);
    ctx.closePath();
    ctx.clip();

    ctx.drawImage(img, (width - logoWidth) / 2, (width - logoWidth) / 2, logoWidth, logoWidth);

    if (!consumer) {
        // match PM4B app
        ctx.strokeStyle = "#e9e9e9";
        ctx.lineWidth = width * 0.015;
        ctx.stroke();
    }

    ctx.restore();
}
